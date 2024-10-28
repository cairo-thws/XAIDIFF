# %%

import torch
import wandb
from xai.XAIPipeline import XAIDDPMPipeline
from xai.XAIGuidance import XAIGuidance
from xai.XAIClassifier import XAIClassifierType, XAIClassifierWrapper
from xai.load_utils import load_classifier
from xai.load_utils import load_dataset, load_sample_images
from xai.utils import FixedSeed, StorageWrapper
import os
import time
from tqdm.auto import tqdm
from xai.torch_utils import lp_dist_pd
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("Starting the experiment")

################################################################################
################################################################################
# %% Define some variables

# Path to the pre-trained diffusion pipeline; may also be a huggingface model name
PATH_PIPELINE = "./XAIDIFF/celeba"
SEED = 0
DEVICE = "cuda"

# Set precsion to torch.float64 for deterministic results
PRECISION = torch.float32

INFERENCE_STEPS = 400
SAMPLE_IMAGE_INDICES = [54, 108, 189, 210]
# images are repeated 4 times
SAMPLE_IMAGE_INDICES = sorted(SAMPLE_IMAGE_INDICES * 4)
SAMPLE_SIZE = len(SAMPLE_IMAGE_INDICES)

# create a new directory for the results including the time in the name as a unique identifier
PATH_RESULTS = os.path.join(os.path.dirname(__file__), "results", f"{time.strftime('%Y%m%d-%H%M%S')}")

CLASSIFIER_CLASS = "mobilenet_v3_large"
CLASSIFIER_TYPE = XAIClassifierType.BinaryMultiLabel
CLASSIFIER_CLASSES = 3

# Skip parts of script for debugging
CALCULATE_METRICS = True

# Experiment Settings
target_classes = [1, 0, 0]
classifier_guidance_scale = 0.2
structural_guidance_scale = 1.0
structural_guidance_lpdistance = 1
adam_step_size = 1

aff_classifier_guidance_scale = 0.2
# %% 0. Prepare the environment
# ensure PATH_RESULTS exists
storage = StorageWrapper(PATH_RESULTS)
storage.ensure_dir()

assert PRECISION in [torch.float32, torch.float64], "Precision must be torch.float32 or torch.float64"

logging.info("Storing results in %s", PATH_RESULTS)

FixedSeed.set_seed(SEED)

################################################################################
################################################################################
# %% 1. Loading the Ground Truth images.
logging.info("1. Loading the ground truth images.")
# First, load the sample images `gt_images`.
# These images will be converted into counterfactuals cf_images with the desired classes.
# The counterfactuals will then be converted back into af_images, without the desired classes.
# gt_images -> cf_images -> af_images

# REPLACE
with FixedSeed(SEED):
    trainset = load_dataset("celeba", (64, 64), download=False)["train"]
    gt_samples, _ = load_sample_images(
        num_samples=SAMPLE_SIZE,
        cls_sub_index=SAMPLE_IMAGE_INDICES,
        device="cpu",
        dataset=trainset,
    )
# /REPLACE

storage.write_image("gt.png", (gt_samples + 1) / 2, nrow=SAMPLE_SIZE)


gt_samples = gt_samples.to(DEVICE).to(PRECISION)

# %% 2. Loading the Classifier
logging.info("2. Loading the classifier.")

# Now we load the classifier.
# The classifier will be used for:
# - extracting the ground truth labels `gt_labels` from the `gt_images`.
# - guiding the transformation gt_images -> cf_images
# - guiding the transformation cf_images -> af_images

# REPLACE
# get channels from gt_images
img_channels = gt_samples.shape[1]

classifier = load_classifier(
    dataset=CLASSIFIER_CLASS,
    num_classes=CLASSIFIER_CLASSES,
    in_channels=img_channels,
    weights_path="pretrained_models/celeba/model.safetensors",
)
# /REPLACE


xai_classifier = XAIClassifierWrapper(classifier=classifier, classifier_type=CLASSIFIER_TYPE)
xai_classifier.initialize(device=DEVICE, precision=PRECISION)

# %% 3. Extract the ground truth labels
logging.info("3. Extract the ground truth labels.")
# We will use the classifier to extract the `gt_labels` from the `gt_images`

with FixedSeed(SEED):
    gt_labels, _ = xai_classifier.predict_labels(gt_samples.clone())


# %% 4. Create and prepare the guidance
logging.info("4. Create and prepare the guidance.")
# Now we create the guidance for the counterfactual generation.
# The guidance will be used to guide the transformation of the gt_images into cf_images.

# get the input shape from the classifier
input_shape = (gt_samples.shape[0], *gt_samples.shape[1:])

cf_guidance = XAIGuidance(
    input_shape=input_shape,
    # (sampling_size,image_channels,image_size,image_size,),
    classifier_wrapper=xai_classifier,
    classifier_target=xai_classifier.extend_target_classes(target_classes, SAMPLE_SIZE),
    classifier_scaling=classifier_guidance_scale,
    structure_ref_images=gt_samples,
    structure_lp_distance=structural_guidance_lpdistance,
    structure_scaling=structural_guidance_scale,
    adam_step_size=adam_step_size,
)

# %% 5. Create the pipeline
logging.info("5. Create the pipeline.")

diffusion_pipeline = XAIDDPMPipeline.from_pretrained(PATH_PIPELINE).to(DEVICE)
diffusion_pipeline.set_guidance(cf_guidance)
diffusion_pipeline.set_precision(PRECISION)


################################################################################
################################################################################
# %% 6. Generate the counterfactuals
logging.info("6. Generate the counterfactuals.")
# Sampling of cf_images
# The pipeline will generate the counterfactuals cf_images from the gt_images
with torch.enable_grad(), FixedSeed(SEED):
    cf_images, cf_samples = diffusion_pipeline(
        batch_size=gt_samples.shape[0],
        num_inference_steps=INFERENCE_STEPS,
        generator=torch.manual_seed(SEED),
        with_grad=True,
        return_dict=False,
    )

# Store the counterfactuals
storage.write_image("cf.png", (cf_samples + 1) / 2, nrow=SAMPLE_SIZE)

wandb.log(
    {
        "gt": wandb.Image((gt_samples + 1) / 2, caption="Ground truth"),
        "cf": wandb.Image((cf_samples + 1) / 2, caption="Counterfactual"),
    }
)
# %% 7. Generate the Affirmatives
logging.info("7. Generate the affirmatives.")
# This takes the counterfactuals cf_images and generates the affirmatives af_images.

# set the af_guidance in the pipeline
af_guidance = cf_guidance.copy(
    structure_ref_images=cf_samples.to(DEVICE),
    classifier_target=gt_labels.float().to(DEVICE),
    classifier_scaling=aff_classifier_guidance_scale,
)

diffusion_pipeline.set_guidance(af_guidance)

with torch.enable_grad(), FixedSeed(SEED):
    af_images, af_samples = diffusion_pipeline(
        batch_size=gt_samples.shape[0],
        num_inference_steps=INFERENCE_STEPS,
        generator=torch.manual_seed(SEED),
        with_grad=True,
        return_dict=False,
    )

# Store the affirmatives
storage.write_image("af.png", (af_samples + 1) / 2, nrow=SAMPLE_SIZE)

################################################################################
################################################################################
# %%
# 8. Ablation
logging.info("8. Ablation of the counterfactuals (generating adversarial samples) without DDPM.")

# generate counterfactuals without closeness
logging.info("8.1 Counterfactuals without closeness.")
cfmc_guidance = cf_guidance.copy(
    structure_scaling=0.0,
)

diffusion_pipeline.set_guidance(cfmc_guidance)

with torch.enable_grad(), FixedSeed(SEED):
    cfmc_images, cfmc_samples = diffusion_pipeline(
        batch_size=gt_samples.shape[0],
        num_inference_steps=INFERENCE_STEPS,
        generator=torch.manual_seed(SEED),
        with_grad=True,
        return_dict=False,
    )

# Store the samples
storage.write_image("cfmc.png", (cfmc_samples + 1) / 2, nrow=SAMPLE_SIZE)

# generate counterfactuals without validity
logging.info("8.2 Counterfactuals without validity.")

cfmv_guidance = cf_guidance.copy(
    classifier_scaling=0.0,
)

diffusion_pipeline.set_guidance(cfmv_guidance)

with torch.enable_grad(), FixedSeed(SEED):
    cfmv_images, cfmv_samples = diffusion_pipeline(
        batch_size=gt_samples.shape[0],
        num_inference_steps=INFERENCE_STEPS,
        generator=torch.manual_seed(SEED),
        with_grad=True,
        return_dict=False,
    )

# Store the samples
storage.write_image("cfmv.png", (cfmv_samples + 1) / 2, nrow=SAMPLE_SIZE)


# generate adversary sampes
logging.info("8.3 Counterfactuals without fidelity (generating adversarial samples) without DDPM.")

ad_samples = cf_guidance.execute_ablation_guidance(gt_samples, step_limit=10000, lr=0.01)
storage.write_image("ad.png", (ad_samples + 1) / 2, nrow=SAMPLE_SIZE)

################################################################################
################################################################################
# %%
# 9. Evaluation
if CALCULATE_METRICS:
    # We will generate multiple metrics to evaluate the quality of the counterfactuals and affirmatives.
    logging.info("9. Evaluation.")
    # 8.1 Calcualte the lp distance between:
    #       gt_images <--> cf_images
    #       gt_images <--> af_images
    #       cf_images <--> af_images

    logging.info("9.1 Calculate the lp distance between gt_images, cf_images, and af_images.")
    gt_samples = gt_samples.cpu()
    cf_samples = cf_samples.cpu()
    ad_samples = ad_samples.cpu()
    cfmc_samples = cfmc_samples.cpu()
    cfmv_samples = cfmv_samples.cpu()
    af_samples = af_samples.cpu()

    lp_gt_cf = lp_dist_pd(gt_samples, cf_samples, 1)
    lp_gt_ad = lp_dist_pd(gt_samples, ad_samples, 1)
    lp_gt_cfmc = lp_dist_pd(gt_samples, cfmc_samples, 1)
    lp_gt_cfmv = lp_dist_pd(gt_samples, cfmv_samples, 1)
    lp_gt_af = lp_dist_pd(gt_samples, af_samples, 1)
    lp_cf_af = lp_dist_pd(cf_samples, af_samples, 1)

    lp_gt_cf_std, lp_gt_cf_mean = torch.std_mean(lp_gt_cf)
    lp_gt_ad_std, lp_gt_ad_mean = torch.std_mean(lp_gt_ad)
    lp_gt_cfmc_std, lp_gt_cfmc_mean = torch.std_mean(lp_gt_cfmc)
    lp_gt_cfmv_std, lp_gt_cfmv_mean = torch.std_mean(lp_gt_cfmv)
    lp_gt_af_std, lp_gt_af_mean = torch.std_mean(lp_gt_af)
    lp_cf_af_std, lp_cf_af_mean = torch.std_mean(lp_cf_af)

    # 8.2 Calculate Bits Per Dim (BPD) on gt_images, cf_images, and af_images
    logging.info("9.2 Calculate Bits Per Dim (BPD) on gt_images, cf_images, and af_images.")
    with FixedSeed(SEED):
        diffusion_pipeline.set_guidance(None)
        nllbpd_gt = diffusion_pipeline.eval_bpd(gt_samples)

        diffusion_pipeline.set_guidance(cf_guidance)
        nllbpd_cf = diffusion_pipeline.eval_bpd(cf_samples)

        diffusion_pipeline.set_guidance(cf_guidance)
        nllbpd_ad = diffusion_pipeline.eval_bpd(ad_samples)

        diffusion_pipeline.set_guidance(cf_guidance)
        nllbpd_cfmc = diffusion_pipeline.eval_bpd(cfmc_samples)

        diffusion_pipeline.set_guidance(cf_guidance)
        nllbpd_cfmv = diffusion_pipeline.eval_bpd(cfmv_samples)

        diffusion_pipeline.set_guidance(af_guidance)
        nllbpd_af = diffusion_pipeline.eval_bpd(af_samples)

    nllbpd_gt_std, nllbpd_gt_mean = torch.std_mean(nllbpd_gt)
    nllbpd_cf_std, nllbpd_cf_mean = torch.std_mean(nllbpd_cf)
    nllbpd_ad_std, nllbpd_ad_mean = torch.std_mean(nllbpd_ad)
    nllbpd_cfmc_std, nllbpd_cfmc_mean = torch.std_mean(nllbpd_cfmc)
    nllbpd_cfmv_std, nllbpd_cfmv_mean = torch.std_mean(nllbpd_cfmv)
    nllbpd_af_std, nllbpd_af_mean = torch.std_mean(nllbpd_af)

    # store all results *raw* in a json file, with as much information as possible
    json = {
        "lp_gt_cf": {"mean": lp_gt_cf_mean.item(), "std": lp_gt_cf_std.item(), "items": lp_gt_cf.tolist()},
        "lp_gt_ad": {"mean": lp_gt_ad_mean.item(), "std": lp_gt_ad_std.item(), "items": lp_gt_ad.tolist()},
        "lp_gt_cfmc": {"mean": lp_gt_cfmc_mean.item(), "std": lp_gt_cfmc_std.item(), "items": lp_gt_cfmc.tolist()},
        "lp_gt_cfmv": {"mean": lp_gt_cfmv_mean.item(), "std": lp_gt_cfmv_std.item(), "items": lp_gt_cfmv.tolist()},
        "lp_gt_af": {"mean": lp_gt_af_mean.item(), "std": lp_gt_af_std.item(), "items": lp_gt_af.tolist()},
        "lp_cf_af": {"mean": lp_cf_af_mean.item(), "std": lp_cf_af_std.item(), "items": lp_cf_af.tolist()},
        "nllbpd_gt": {"mean": nllbpd_gt_mean.item(), "std": nllbpd_gt_std.item(), "items": nllbpd_gt.tolist()},
        "nllbpd_cf": {"mean": nllbpd_cf_mean.item(), "std": nllbpd_cf_std.item(), "items": nllbpd_cf.tolist()},
        "nllbpd_ad": {"mean": nllbpd_ad_mean.item(), "std": nllbpd_ad_std.item(), "items": nllbpd_ad.tolist()},
        "nllbpd_cfmc": {"mean": nllbpd_cfmc_mean.item(), "std": nllbpd_cfmc_std.item(), "items": nllbpd_cfmc.tolist()},
        "nllbpd_cfmv": {"mean": nllbpd_cfmv_mean.item(), "std": nllbpd_cfmv_std.item(), "items": nllbpd_cfmv.tolist()},
        "nllbpd_af": {"mean": nllbpd_af_mean.item(), "std": nllbpd_af_std.item(), "items": nllbpd_af.tolist()},
    }
    storage.write_json(
        "metrics.json",
        json,
    )
    wandb.log(json)

# %% 10. Plotting
logging.info("10. Validate the results of the classifier.")


from xai.XAIClassifier import XaiGenerationType

# Validate the results of the classifier
with FixedSeed(SEED):
    gt_samples_annotated, gt_accuracy = xai_classifier.validate(
        samples=gt_samples,
        target_classes=xai_classifier.extend_target_classes(target_classes, SAMPLE_SIZE),
        original_classes=gt_labels,
        experiment_target_classes=xai_classifier.extend_target_classes(target_classes, SAMPLE_SIZE),
        generation_type=XaiGenerationType.Counterfactual,
    )
    cf_samples_annotated, cf_accuracy = xai_classifier.validate(
        samples=cf_samples,
        target_classes=xai_classifier.extend_target_classes(target_classes, SAMPLE_SIZE),
        original_classes=gt_labels,
        experiment_target_classes=xai_classifier.extend_target_classes(target_classes, SAMPLE_SIZE),
        generation_type=XaiGenerationType.Counterfactual,
    )
    af_samples_annotated, af_accuracy = xai_classifier.validate(
        samples=af_samples,
        target_classes=gt_labels,
        original_classes=xai_classifier.extend_target_classes(target_classes, SAMPLE_SIZE),
        experiment_target_classes=xai_classifier.extend_target_classes(target_classes, SAMPLE_SIZE),
        generation_type=XaiGenerationType.Affirmative,
    )
    # Ablation Values
    ad_samples_annotated, ad_accuracy = xai_classifier.validate(
        samples=ad_samples,
        target_classes=xai_classifier.extend_target_classes(target_classes, SAMPLE_SIZE),
        original_classes=gt_labels,
        experiment_target_classes=xai_classifier.extend_target_classes(target_classes, SAMPLE_SIZE),
        generation_type=XaiGenerationType.Counterfactual,
    )
    cfmc_samples_annotated, cfmc_accuracy = xai_classifier.validate(
        samples=cfmc_samples,
        target_classes=xai_classifier.extend_target_classes(target_classes, SAMPLE_SIZE),
        original_classes=gt_labels,
        experiment_target_classes=xai_classifier.extend_target_classes(target_classes, SAMPLE_SIZE),
        generation_type=XaiGenerationType.Counterfactual,
    )
    cfmv_samples_annotated, cfmv_accuracy = xai_classifier.validate(
        samples=cfmv_samples,
        target_classes=xai_classifier.extend_target_classes(target_classes, SAMPLE_SIZE),
        original_classes=gt_labels,
        experiment_target_classes=xai_classifier.extend_target_classes(target_classes, SAMPLE_SIZE),
        generation_type=XaiGenerationType.Counterfactual,
    )

raw_samples = torch.stack(
    [gt_samples.cpu(), cf_samples.cpu(), af_samples.cpu(), ad_samples.cpu(), cfmc_samples.cpu(), cfmv_samples.cpu()],
    dim=0,
)

# Plot gt_images, cf_images, af_images in a grid
annotated_samples = torch.stack(
    [
        gt_samples_annotated.cpu(),
        cf_samples_annotated.cpu(),
        af_samples_annotated.cpu(),
        ad_samples_annotated.cpu(),
        cfmc_samples_annotated.cpu(),
        cfmv_samples_annotated.cpu(),
    ],
    dim=0,
)

storage.write_tensor("samples_annotated.pt", annotated_samples)
storage.write_tensor("samples_raw.pt", raw_samples)
storage.write_image_with_row_text(
    "results.png",
    (annotated_samples + 1) / 2,
    row_texts=[
        "Source",
        "Counterfactual",
        "Affirmative",
        "Adversarial",
        "Counterfactual-closeness",
        "Counterfactual-validity",
    ],
)
wandb.log(
    {
        "results": wandb.Image((annotated_samples + 1) / 2, caption="Results"),
    }
)
storage.write_json(
    "hyperparameters.json",
    {
        "classifier_guidance_scale": classifier_guidance_scale,
        "adam_step_size": adam_step_size,
    },
)
