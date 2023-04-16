instances = [["mask","no"], ["mask","no"], ["no","mask"]]

unique_list = list(set(tuple(i) for i in instances))
unique_list = [list(elem) for elem in unique_list]
print(unique_list)