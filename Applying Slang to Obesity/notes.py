   elif args.bnn_mnist_table:
        if RUN_EXPERIMENT:
            ask_confirm(LONG)
            call("python " + path_to("python/submitters/mnist/submit_mnist_experiment.py"))
            call("python " + path_to("python/run_and_plot_scripts/mnist/make_table_3.py"))





##it seems like this is what is running the experiment in main.py
