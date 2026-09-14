from time import perf_counter

from quantile_analysis import run_quantile_analysis


# -------------------------
# COMPLETE FACTOR SELECTION LAYER
def run_pipeline():
    start_time = perf_counter()

    try:
        return run_quantile_analysis()
    finally:
        elapsed_seconds = perf_counter() - start_time
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            "Factor Selection pipeline time: "
            f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
        )


if __name__ == "__main__":
    run_pipeline()
