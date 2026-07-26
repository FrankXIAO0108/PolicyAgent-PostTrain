import json
from pathlib import Path


INPUT_FILE = Path(
    "reports/failure_analysis/failure_summary.json"
)

OUTPUT_DIR = Path(
    "reports/failure_analysis"
)


TAXONOMY = {

    "59": {
        "category": "golden_user_mismatch",
        "description":
            "User intent and evaluation golden mismatch. "
            "Agent behavior should be analyzed separately.",
        "severity": "medium"
    },

    "98": {
        "category": "scope_confirmation_failure",
        "description":
            "Agent failed to preserve user's requested scope "
            "and performed broader action than requested.",
        "severity": "high"
    },

    "95": {
        "category": "variant_understanding_failure",
        "description":
            "Agent misunderstood product variant constraints "
            "and selected incorrect exchange option.",
        "severity": "high"
    },

    "107": {
        "category": "policy_violation",
        "description":
            "Agent action violates business policy even "
            "though tool execution succeeded.",
        "severity": "high"
    }

}



def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        failures = json.load(f)


    results = []


    for item in failures:

        task_id = str(
            item["task_id"]
        )


        result = {

            "task_id": task_id,

            "reward": item["reward"],

            "message_count":
                item["message_count"],

            "tool_calls":
                item["tool_calls"],

            "taxonomy":
                TAXONOMY.get(
                    task_id,
                    {
                        "category": "unknown",
                        "description": "",
                        "severity": "unknown"
                    }
                )
        }


        results.append(result)



    json_path = (
        OUTPUT_DIR /
        "failure_taxonomy.json"
    )


    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )



    md_path = (
        OUTPUT_DIR /
        "failure_taxonomy.md"
    )


    with open(
        md_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# Failure Taxonomy\n\n"
        )


        for r in results:

            t = r["taxonomy"]

            f.write(
                f"## Task {r['task_id']}\n\n"
            )

            f.write(
                f"- Category: `{t['category']}`\n"
            )

            f.write(
                f"- Severity: `{t['severity']}`\n"
            )

            f.write(
                f"- Description: {t['description']}\n\n"
            )



    print(
        "Generated failure taxonomy"
    )



if __name__ == "__main__":
    main()