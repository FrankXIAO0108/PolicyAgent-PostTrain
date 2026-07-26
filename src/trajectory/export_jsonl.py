import json

from pathlib import Path


from .parser import parse_task_trajectory



EXPERIMENT_DIR = Path(
    "experiments/"
    "20260722_110504_retail_baseline20_trial1_deepseek"
)


OUTPUT_FILE = Path(
    "data/trajectory/"
    "retail_baseline20_trial1.jsonl"
)



def main():


    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    task_dirs = sorted(

        [
            p
            for p in EXPERIMENT_DIR.iterdir()

            if p.is_dir()
            and p.name.startswith(
                "task_"
            )
        ],

        key=lambda x:
            int(
                x.name.split("_")[1]
            )
    )



    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:


        count = 0


        for task_dir in task_dirs:


            try:

                traj = parse_task_trajectory(
                    task_dir
                )


                f.write(

                    json.dumps(
                        traj.to_dict(),
                        ensure_ascii=False
                    )
                    +
                    "\n"
                )


                count += 1


                print(
                    f"Exported {task_dir.name}"
                )


            except Exception as e:


                print(
                    f"Failed {task_dir.name}: {e}"
                )



    print(
        f"\nFinished: {count} trajectories"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )



if __name__ == "__main__":

    main()