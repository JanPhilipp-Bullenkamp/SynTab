def export_labels(labels, filepath):
    with open(filepath, "w") as f:
        f.write("# +-----------------------------------------------------+\n")
        f.write("# | txt file with labels per index                      |\n")
        f.write("# +-----------------------------------------------------+\n")
        f.write("# | Format: index label                                 |\n")
        f.write("# +-----------------------------------------------------+\n")
        for index, label in enumerate(labels):
            if isinstance(label, list):
                label = " ".join(str(l) for l in label)
            f.write(f"{index} {label}\n")
