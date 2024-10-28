from enum import Enum
import torch


class XAIClassifierType(Enum):
    # multi-class or binary-multi-label
    MultiClass = 0
    BinaryMultiLabel = 1


class XaiGenerationType(Enum):
    Counterfactual = 0
    Affirmative = 1


class XAIClassifierWrapper(object):
    def __init__(self, classifier: torch.nn.Module, classifier_type: XAIClassifierType):
        self.classifier = classifier
        self.classifier_type = classifier_type

    def initialize(self, device: torch.device = None, precision=None):
        if device is not None:
            self.classifier.to(device)
        if precision is not None:
            self.classifier = self.classifier.to(precision)
        self.classifier.eval()

    def extend_target_classes(self, sampling_target_class, sample_size):
        sampling_target_class = str(sampling_target_class)
        if self.classifier_type == XAIClassifierType.MultiClass:
            return (torch.ones(sample_size, dtype=torch.long) * eval(sampling_target_class)).to(self.classifier_device)
        elif self.classifier_type == XAIClassifierType.BinaryMultiLabel:
            return (
                torch.tensor(eval(sampling_target_class), dtype=torch.float32)
                .repeat((sample_size, 1))
                .to(self.classifier_device)
            )
        else:
            NotImplementedError

    @property
    def loss_fn(self):
        if self.classifier_type == XAIClassifierType.MultiClass:
            return torch.nn.CrossEntropyLoss(reduction="sum")
        elif self.classifier_type == XAIClassifierType.BinaryMultiLabel:
            return torch.nn.BCEWithLogitsLoss(reduction="sum")
        else:
            NotImplementedError

    @property
    def label_dtype(self):
        if self.classifier_type == XAIClassifierType.MultiClass:
            return torch.long
        elif self.classifier_type == XAIClassifierType.BinaryMultiLabel:
            return torch.float32
        else:
            NotImplementedError

    @property
    def label_dtype(self):
        if self.classifier_type == XAIClassifierType.MultiClass:
            return torch.long
        elif self.classifier_type == XAIClassifierType.BinaryMultiLabel:
            return torch.float
        else:
            NotImplementedError

    @property
    def classifier_device(self):
        return list(self.classifier.parameters())[0].device

    def predict_labels(self, images: torch.Tensor):
        logits = self.classifier(images.to(self.classifier_device)).cpu()
        label_pred = None
        if self.classifier_type == XAIClassifierType.MultiClass:
            label_pred = logits.argmax(dim=1)
        elif self.classifier_type == XAIClassifierType.BinaryMultiLabel:
            label_pred = (logits.to(torch.float32).sigmoid() >= 0.5).long()
        else:
            raise NotImplementedError
        return label_pred, logits

    def validate(
        self,
        samples: torch.Tensor,
        original_classes: torch.Tensor,
        target_classes: torch.Tensor,
        # The classes of the counterfactuals
        experiment_target_classes: torch.Tensor,
        generation_type: XaiGenerationType,
    ):
        """
        Annotates the samples with stripes to indicate the result of the classifier.
        Colors:
            - Red: Counterfactuals that are classified as the target class
        """

        samples = samples.clone().detach().to(self.classifier_device)
        label_pred, logits = self.predict_labels(samples)

        # Grayscale to RGB
        if samples.shape[1] == 1:
            samples = torch.repeat_interleave(samples, 3, dim=1)

        # The color indicates the result of the classifier
        color_map = {
            # (equals_target, equals_orig, generation_type)
            (True, False, XaiGenerationType.Counterfactual): 1,
            (True, False, XaiGenerationType.Affirmative): 2,
            (False, True, XaiGenerationType.Counterfactual): 2,
            (False, True, XaiGenerationType.Affirmative): 1,
        }

        # Add stripes depending on the result and generation type
        samples[:, :, :3, :] = 0
        for idx, label in enumerate(label_pred):
            equals_target = torch.equal(label, target_classes[idx].cpu().long())
            equals_orig = torch.equal(label, original_classes[idx].cpu().long())

            color = color_map.get((equals_target, equals_orig, generation_type), 0)
            samples[idx, color, :3, :] = 1

        ## Calculate accuracy to the initial target class of the XAI process
        if self.classifier_type == XAIClassifierType.MultiClass:
            accuracy = logits.softmax(dim=1)[
                range(0, len(experiment_target_classes)), experiment_target_classes.int().cpu()
            ].mean()
        elif self.classifier_type == XAIClassifierType.BinaryMultiLabel:
            label_probs = logits.sigmoid()
            if sum(experiment_target_classes) == 1:  # Single label
                accuracy = label_probs[:, torch.argmax(experiment_target_classes.int().cpu())].mean()
            else:  # Multi label
                accuracy = label_probs.mean(dim=1).mean()
        else:
            NotImplementedError("Classifier type not implemented.")

        return samples, accuracy

    def __call__(self, *args, **kwargs):
        return self.classifier(*args, **kwargs)
