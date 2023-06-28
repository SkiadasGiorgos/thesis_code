import pickle

save_path = "landmarks_new.pkl"

objects = []
with (open(save_path, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
            # objects.append(torch.load(save_path,map_location=torch.device('cpu')))
        except EOFError:
            break
print(objects)