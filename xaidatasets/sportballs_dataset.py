import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class SportBallsDataset(Dataset):
    def __init__(self):
        "Custom images to place on the white canvas"
        self.baseball = Image.open("xaidatasets/sportballs/baseball.png", "r")
        self.basketball = Image.open("xaidatasets/sportballs/basketball.png", "r")
        self.volleyball = Image.open("xaidatasets/sportballs/volleyball.png", "r")
        self.soccerball = Image.open("xaidatasets/sportballs/soccerball.png", "r")
        self.tennisball = Image.open("xaidatasets/sportballs/tennisball.png", "r")
        self.sportballs = [self.baseball, self.basketball, self.volleyball, self.soccerball, self.tennisball]

        [i.load() for i in self.sportballs]  # Resolves lazy loading of PIL Images for multiple workers in data loader

        self.IMAGE_SIZE = 64
        self.CANVAS_SIZE = 128
        self.N_OBJECTS = 3

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        )

    def load_dataset(self):
        "Compatibility with huggingface data sets"
        return {"train": self}

    def __len__(self):
        "Custom length for data set"
        return 10000

    def __getitem__(self, idx):
        background = Image.new("RGBA", (self.CANVAS_SIZE, self.CANVAS_SIZE), (255, 255, 255, 255))

        class_idx = torch.randint(0, len(self.sportballs) - 1, (self.N_OBJECTS,), generator=torch.manual_seed(idx))
        pos_idx = torch.randint(
            0, self.CANVAS_SIZE - self.IMAGE_SIZE, (self.N_OBJECTS, 2), generator=torch.manual_seed(idx)
        )
        rotation = torch.randint(0, 360, (self.N_OBJECTS,), generator=torch.manual_seed(idx))
        size = torch.randint(3, 9, (self.N_OBJECTS,), generator=torch.manual_seed(idx)) / 10.0

        for cls, pos, rot, s in zip(class_idx, pos_idx, rotation, size):
            obj = self.sportballs[cls].rotate(rot).resize((int(self.IMAGE_SIZE * s), int(self.IMAGE_SIZE * s)))
            background.paste(obj, pos.tolist(), obj)

        background = background.convert("RGB")

        if self.transform:
            background = self.transform(background)

        return background, int(3 in class_idx)  # 3 = Soccer Ball
