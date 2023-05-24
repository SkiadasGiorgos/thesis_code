import splitfolders

def split_files(data_location, output_location,seed):
    splitfolders.ratio(data_location, # The location of dataset
                   output=output_location, # The output location
                   seed=seed, # The number of seed
                   ratio=(.7, .2, .1), # The ratio of splited dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )