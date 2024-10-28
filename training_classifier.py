import argparse
from tqdm.auto import tqdm
from xaidatasets.sportballs_dataset import SportBallsDataset
from xai.load_utils import load_classifier, load_dataset, load_huggingface_dataset
from xai.XAIClassifier import XAIClassifierType, XAIClassifierWrapper
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch

parser = argparse.ArgumentParser(description="Diffusion model training.")
parser.add_argument("--dataset", type=str, help="Data set string")
parser.add_argument("--imgchannels", type=int, default=3, required=False, help="Image channels")
parser.add_argument("--classes", type=int, required=True, help="Classes")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAME = args.dataset
IMAGE_SIZE = 64
IMAGE_CHANNELS = args.imgchannels
CLASSIFIER_CLASSES = args.classes
CLASSIFIER_TYPE = XAIClassifierType.MultiClass
CLASSIFIER_CLASS = "mobilenet_v3_small"  # mobilenet_v3_small, mobilenet_v3_large, squeeze_net, efficientnet_b3
SEED = 1234

TRAINING_LR = 1e-4
TRAINING_USE_FP16 = True
TRAINING_BATCH_SIZE = 32
TRAINING_EPOCHS = 10

DATALOADER_NUM_WORKERS = 8

# %% Load the data set
if DATASET_NAME == "sportballs":
    datasets = SportBallsDataset().load_dataset()
    collate_fn = None
else:
    datasets = load_huggingface_dataset(DATASET_NAME, (IMAGE_SIZE, IMAGE_SIZE), IMAGE_CHANNELS)
    collate_fn = lambda i: list(torch.utils.data.default_collate(i).values())


if "validation" in datasets.keys():
    valset, trainset = datasets["validation"], datasets["train"]
else:
    valset, trainset = torch.utils.data.random_split(
        datasets["train"], [0.2, 0.8], torch.Generator().manual_seed(SEED)
    )

trainloader = DataLoader(
    trainset,
    batch_size=TRAINING_BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=DATALOADER_NUM_WORKERS,
    persistent_workers=True,
    collate_fn=collate_fn,
)
valloader = DataLoader(
    valset,
    batch_size=TRAINING_BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=DATALOADER_NUM_WORKERS,
    persistent_workers=True,
    collate_fn=collate_fn,
)

# %% Initialize the model
classifier = load_classifier(
    dataset=CLASSIFIER_CLASS,
    num_classes=CLASSIFIER_CLASSES,
    in_channels=IMAGE_CHANNELS,
)
xai_classifier = XAIClassifierWrapper(classifier=classifier, classifier_type=CLASSIFIER_TYPE)
xai_classifier.initialize(device=DEVICE)
# %% Define optimizer and loss for training
optimizer = torch.optim.AdamW(classifier.parameters(), lr=TRAINING_LR)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=(len(trainloader) * TRAINING_EPOCHS),
)


loss_fn = xai_classifier.loss_fn
label_dtype = xai_classifier.label_dtype

# %% Initialize accelerator and wandb logging
accelerator = Accelerator(
    mixed_precision="fp16" if TRAINING_USE_FP16 else "no",
    gradient_accumulation_steps=1,
    log_with="wandb",
    project_dir="XAIDIFFClassifierTraining",
)

(
    classifier,
    optimizer,
    trainloader,
    valloader,
    lr_scheduler,
    loss_fn,
) = accelerator.prepare(classifier, optimizer, trainloader, valloader, lr_scheduler, loss_fn)

accelerator.init_trackers("XAIDIFFTraining", init_kwargs={"entity": "anonymous"})

# %% Print model information
model_parameter_count = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
accelerator.log({"classifier_parameter_count": model_parameter_count})
print("Classifier Parameters:     ", f"{model_parameter_count:,}")

# %% Training

for epoch in tqdm(range(TRAINING_EPOCHS), desc="Epoch"):
    classifier.train()
    for batch in tqdm(trainloader, desc="Batch"):
        images, labels = batch
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = classifier(images)
        loss = loss_fn(outputs, labels.to(label_dtype))
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
    classifier.eval()
    with torch.no_grad():
        val_loss = 0
        val_correct = 0
        for batch in valloader:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = classifier(images)
            val_loss += loss_fn(outputs, labels.to(label_dtype)).item()
            val_correct += (outputs.argmax(dim=1) == labels).float().sum()
        val_loss /= len(valloader)
        val_acc = val_correct / valloader.total_dataset_length
        accelerator.log({"val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch}, val_loss: {val_loss:.4f}, val_acc: {val_acc*100:.2f} %")

    accelerator.save_model(
        classifier,
        f"{accelerator.trackers[0].run.dir}/",
    )

accelerator.end_training()
