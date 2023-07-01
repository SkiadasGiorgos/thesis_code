import pickle

save_path = "landmarks_new.pkl"

objects = []
with (open(save_path, "rb")) as openfile:
    
    while True:
        try:
            objects.append(pickle.load(openfile))
            dir = pickle.load(openfile)        
            # objects.append(torch.load(save_path,map_location=torch.device('cpu')))
        except EOFError:
            break
# print(objects)

if '/nas2/ckoutlis/DataStorage/vggface2/data/test/n001857/036801.jpg' in dir:
    print(dir.get('/nas2/ckoutlis/DataStorage/vggface2/data/test/n001857/0368_01.jpg'))
else: 
    print(None)