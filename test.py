# instances = [["mask","no"], ["mask","no"], ["no","mask"]]

# unique_list = list(set(tuple(i) for i in instances))
# unique_list = [list(elem) for elem in unique_list]
# print(unique_list)


def format_line(line):
    parts = line.strip().split()
    my_str = f"{parts[0]}=={parts[1]}"
    return my_str


with open("correct_installs.txt", "r") as input_file:
    lines = input_file.readlines()

formatted_lines = [format_line(line) for line in lines]

with open("correct_installs_fixed.txt", "w") as output_file:
    output_file.write("\n".join(formatted_lines))
