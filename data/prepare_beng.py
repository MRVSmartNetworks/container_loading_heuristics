#!/usr/bin/env python3

import os
import sys

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    dirname = os.path.join(script_dir, "BENG")

    for fname in os.listdir(dirname):
        if os.path.splitext(fname)[-1] == ".ins2D":
            new_dir = os.path.join(
                script_dir, os.path.splitext(os.path.basename(fname))[0]
            )
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)

            fi = open(os.path.join(new_dir, "items.csv"), "w")
            fi.write(
                "id_item,length,width,height,weight,nesting_height,stackability_code,forced_orientation,max_stackability\n"
            )
            with open(os.path.join(dirname, fname), "r") as f:
                for i, line in enumerate(f):
                    ln = line.rstrip()
                    if i == 0:
                        n_items = int(ln)
                    elif i == 1:
                        # Truck info
                        with open(
                            os.path.join(new_dir, "vehicles.csv"), "w"
                        ) as fv:
                            fv.write(
                                "id_truck,length,width,height,max_weight,max_weight_stack,cost,max_density\n"
                            )
                            val_line = [int(x) for x in ln.split()]
                            assert len(val_line) == 2
                            fv.write(
                                f"V0,{val_line[0]},{val_line[1]},1,100000,100000,1,100000\n"
                            )
                            fv.close()
                    else:
                        vals_line = [int(x) for x in ln.split()]
                        assert len(vals_line) == 6
                        item_id = f"I{str(vals_line[0]).rjust(4, '0')}"
                        fi.write(
                            f"{item_id},{vals_line[1]},{vals_line[2]},1,1,0,{i},n,1\n"
                        )
                f.close()
            fi.close()

    print("Files were correctly created!")
