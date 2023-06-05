from torchvision import transforms

# Values were found here https://huggingface.co/google/vit-base-patch16-224-in21k
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]

def create_transforms(train=True):
  if train:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224), antialias=True),
        transforms.Normalize(mean=mean,
                             std=std
                             ),
    ])
  else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std
                             ),
        transforms.Resize((224,224), antialias=True),
    ])

  return transform