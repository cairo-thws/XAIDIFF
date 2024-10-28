import torch
import logging
from tqdm import tqdm
from xai.XAIClassifier import XAIClassifierWrapper

log = logging.getLogger(__name__)

WANDB_LOGGING = True


class XAIGuidance:
    def __init__(
        self,
        input_shape,
        classifier_wrapper: XAIClassifierWrapper,
        classifier_target,
        classifier_scaling,
        structure_ref_images,
        structure_lp_distance,
        structure_scaling,
        adam_step_size,
    ):
        self.input_shape = input_shape
        self.classifier_wrapper = classifier_wrapper
        self.classifier_target = classifier_target.to(classifier_wrapper.label_dtype)
        self.classifier_scaling = classifier_scaling
        self.classifier_loss_fn = self.classifier_wrapper.loss_fn
        self.structure_ref_images = structure_ref_images.detach().clone()
        self.structure_lp_distance = structure_lp_distance
        self.structure_scaling = structure_scaling
        self.adam = ADAMGradientStabilization(
            input_shape=input_shape, device=list(classifier_wrapper.classifier.parameters())[0].device
        )
        self.adam_step_size = adam_step_size

        if structure_lp_distance == 1:
            self.structure_loss_fn = torch.nn.L1Loss(reduction="sum")
        elif structure_lp_distance == 2:
            self.structure_loss_fn = torch.nn.MSELoss(reduction="sum")
        else:
            NotImplementedError

    def __call__(self, xt, xzeropred):
        return self.get_diffusion_guidance(xt, xzeropred)

    @torch.enable_grad()
    def get_diffusion_guidance(self, xt, xzeropred):
        # This method is drastically impacted by precision errors at the torch.autograd.grad call
        # This can lead to non-deterministic results.

        # Check for correct input scaling [-1,1]
        assert xzeropred.min() < 0 and xzeropred.min() >= -1 and xzeropred.max() <= 1

        # Calculate classifier loss
        logits = self.classifier_wrapper.classifier(xzeropred)
        classifier_loss = -self.classifier_loss_fn(logits, self.classifier_target) / logits.numel()

        # Calculate distance loss
        structure_loss = (
            -self.structure_loss_fn(self.structure_ref_images, xzeropred) / self.structure_ref_images.numel()
        )

        # Combine losses and calculate one gradient over weighted loss terms
        combined_loss = (self.classifier_scaling * classifier_loss) + (self.structure_scaling * structure_loss)

        if WANDB_LOGGING:
            import wandb

            wandb.log(
                {
                    "sloss": structure_loss.detach().cpu(),
                    "sloss_scaled": self.structure_scaling * structure_loss.detach().cpu(),
                    "closs": classifier_loss.detach().cpu(),
                    "closs_scaled": self.classifier_scaling * classifier_loss.detach().cpu(),
                    "cbloss": combined_loss.detach().cpu(),
                }
            )
        target_grad = torch.autograd.grad(combined_loss, xt)[0]
        #! TODO: autograd is not deterministic at the moment

        target_grad = self.adam_step_size * self.adam(target_grad)
        return target_grad

    @torch.enable_grad()
    def execute_ablation_guidance(self, xt: torch.Tensor, step_limit=1000, lr=0.085):
        xt = xt.to(self.classifier_wrapper.classifier_device)
        counterfactuals = xt.clone()

        step = 0
        progress_bar = tqdm(desc="Executing Ablation", leave=False)
        while step < step_limit:
            label_pred, _ = self.classifier_wrapper.predict_labels(counterfactuals)
            label_pred = label_pred.to(self.classifier_target.device)
            # if all labels are correct, abort
            if torch.all(label_pred == self.classifier_target):
                log.info(f"Ablation step {step}: All labels are equal to the target labels.")
                break

            counterfactuals = counterfactuals.detach().requires_grad_(True)
            logits = self.classifier_wrapper.classifier(counterfactuals)
            validity_loss = -self.classifier_loss_fn(logits, self.classifier_target)

            closeness_loss = self.structure_loss_fn(xt, counterfactuals)

            combined_grad = torch.autograd.grad(validity_loss - closeness_loss, counterfactuals)[0]

            # set combined_grad to 0 if the label is already correct
            combined_grad[label_pred == self.classifier_target] = 0

            counterfactuals = counterfactuals + lr * combined_grad
            counterfactuals = counterfactuals.clamp(-1, 1)
            step += 1
            progress_bar.update(1)

        return counterfactuals

    # copy constructor; overwrites all parameters with new values if given, otherwise keeps the old values
    def copy(self, **kwargs):
        return XAIGuidance(
            input_shape=kwargs.get("input_shape", self.input_shape),
            classifier_wrapper=kwargs.get("classifier_wrapper", self.classifier_wrapper),
            classifier_target=kwargs.get("classifier_target", self.classifier_target),
            classifier_scaling=kwargs.get("classifier_scaling", self.classifier_scaling),
            structure_ref_images=kwargs.get("structure_ref_images", self.structure_ref_images),
            structure_lp_distance=kwargs.get("structure_lp_distance", self.structure_lp_distance),
            structure_scaling=kwargs.get("structure_scaling", self.structure_scaling),
            adam_step_size=kwargs.get("adam_step_size", self.adam_step_size),
        )


class ADAMGradientStabilization:
    def __init__(self, input_shape, device, beta_1=0.9, beta_2=0.999, eps=1e-8):
        super().__init__()
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.m = torch.zeros(input_shape).to(device)
        self.v = torch.zeros(input_shape).to(device)
        self.step = 1

    def __call__(self, classifier_gradient):
        m = self.beta_1 * self.m + (1 - self.beta_1) * classifier_gradient
        self.m = m
        v = self.beta_2 * self.v + (1 - self.beta_2) * torch.square(classifier_gradient)
        self.v = v
        m_hat = m / (1 - (self.beta_1**self.step))
        v_hat = v / (1 - (self.beta_2**self.step))
        self.step += 1
        return m_hat / (torch.sqrt(v_hat) + self.eps)
