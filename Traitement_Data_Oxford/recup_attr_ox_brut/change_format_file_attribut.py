file = 'oxford102_attributs.txt'

target_colors = ["red", "blue", "pink", "yellow", "purple", "white", "orange", "violet", "green"] #liste des couleurs

f = open(file)
content = f.read()
lines = content.split('\n')
filenames = []
attributes_list = []
for line in lines:

    parts = line.split(' ')
    filenames.append(parts[0]) #nom des fichiers
    attributes_list.append(parts[1:]) #nom de l'attributs du fichier
f.close()

output_file = "/home/ityt/Documents/Hicham/M2/MLA/Traitement_base_oxford/test_true.txt"

results = []
results.append("8189") #nombre d'images
results.append("red blue pink yellow purple white orange violet green") #deuxieme ligne où on mets les différents attributs

for i in range(len(filenames)):

    modified_line = filenames[i] + ' '
    for color in target_colors:
        if color in attributes_list[i]: #on fait le formatage
            modified_line += '1 '
        else:
            modified_line += '-1 '
    results.append(modified_line.strip())

with open(output_file, "w") as f:
    f.write("\n".join(results))
    print("done")