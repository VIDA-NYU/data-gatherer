from data_gatherer.orchestrator import Orchestrator
import time

if __name__ == "__main__":
    t0 = time.time()
    config_path = 'data_gatherer/config/config.json'  # Config with input file details

    # Initialize driver and orchestrator using confi
    orchestrator = Orchestrator(config_path)

    # Process URLs and print the results
    results = orchestrator.run()

    # Output the results
    # print(results)

    # Print the time taken
    print(f"Time taken: {time.time() - t0:.2f} seconds")