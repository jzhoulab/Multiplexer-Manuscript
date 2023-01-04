import torch 
###The following code zero's out the beluga predictions that correspond to the reference

size = 2500

beluga = torch.load("../../predictions/Beluga_predictions").reshape(size, 2000, 4, 2002) #shape 2500x8000x2002
ref_indices = torch.load("../../../data/comparison/inputs/ref_indices") #2500x2000 gives the indices to ignore


###Manually reshape beluga predictions into 2500, 3, 2000, 2002 then -> 2500, 2000, 2002
new_beluga = torch.zeros((size, 2000, 3, 2002))

for i in range(size):
    for j in range(2000):
        index = ref_indices[i][j]
        
        if index == 0:
            a,b,c = 1,2,3
            new_beluga[i, j, 0, :] = beluga[i, j, a, :]
            new_beluga[i, j, 1, :] = beluga[i, j, b, :]
            new_beluga[i, j, 2, :] = beluga[i, j, c, :]
            
        if index == 1:
            a,b,c = 0,2,3
            new_beluga[i, j, 0, :] = beluga[i, j, a, :]
            new_beluga[i, j, 1, :] = beluga[i, j, b, :]
            new_beluga[i, j, 2, :] = beluga[i, j, c, :]
            
        if index == 2:
            a,b,c = 0,1,3
            new_beluga[i, j, 0, :] = beluga[i, j, a, :]
            new_beluga[i, j, 1, :] = beluga[i, j, b, :]
            new_beluga[i, j, 2, :] = beluga[i, j, c, :]
            
        if index == 3:
            a,b,c = 0,1,2
            new_beluga[i, j, 0, :] = beluga[i, j, a, :]
            new_beluga[i, j, 1, :] = beluga[i, j, b, :]
            new_beluga[i, j, 2, :] = beluga[i, j, c, :]
            
            
            
torch.save(new_beluga, "../zerod_data/zerod_beluga")
            