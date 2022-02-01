import json
import itertools

path = "C:/Users/user/pt_cloud_data/"
sync_ply = "sync.ply"
e57item =[path+ "atom- 005.e57"]

names = []
for i in range(1, 74, 5):
    ar = 3 - len(str(i))
    if ar==2:
        nl = '00'
    if ar==1:
        nl = '0'
    
    
    name =  path+"atom- "+nl+str(i)+".e57"
    names.append(name)
print(names)


task_dict = {"path": path, "sync_ply":sync_ply, "e57item":e57item, "e57iter": names}


with open("task.json", "w") as read_file:
    json.dump(task_dict, read_file)
