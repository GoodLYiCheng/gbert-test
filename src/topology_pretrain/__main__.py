from .cli import main


# multiprocessing "spawn" imports the main module in every worker.  Calling
# the CLI unconditionally here recursively starts another trainer/pool and
# leaves all workers idle on Linux.  Only the actual CLI process may run it.
if __name__ == "__main__":
    main()
