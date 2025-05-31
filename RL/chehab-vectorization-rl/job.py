import subprocess
import re
from model import CustomFeaturesExtractor


SBATCH_SCRIPT = "./exec_job.sh"


def main():

    job_id = submit_sbatch()
    if not job_id:
        print("Job submission failed. Exiting...")
        return


def submit_sbatch():
    """Submit the SLURM job and return the job ID."""
    try:
        result = subprocess.run(
            ["sbatch", SBATCH_SCRIPT],
            capture_output=True,
            text=True,
            check=True
        )
        job_id_match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if job_id_match:
            job_id = job_id_match.group(1)
            print(f"Job submitted successfully. Job ID: {job_id}")
            return job_id
        else:
            raise ValueError("Could not parse job ID from sbatch output.")
    except subprocess.CalledProcessError as e:
        print("Failed to submit job:", e.stderr)
        return None

if __name__ == "__main__":
    main()
