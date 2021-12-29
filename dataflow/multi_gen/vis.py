def set_vis_args(datactr):
    from visualizer.OpencvVis import CVis
    datactr.vis = CVis(grab_pcl_fuc=datactr.get_data,
                            switch_model_func=datactr.switch_model,
                            switch_dataset_func=datactr.switch_dataset,
                            switch_pcl_func=datactr.switch_pcl,
                            _3d=False
                            )
    datactr.config.Set_datactr_func=None
    datactr.config.Set_exec_func = datactr.vis.run_pcl_dsiplay_process
    datactr.set()
    pass