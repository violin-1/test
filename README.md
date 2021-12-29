有数据控制模块(Dataflow/Datacontroller)
    该模块中:
    外置初始化接口，Set_datactr_func
    执行模块接口，exec_func
    代加载数据类型，datatsetstype []
    数据显示参数设置接口，datasets_vis_arg_func []
    数据操作设备:device
    内嵌模型控制
常见的数据操作分为：
    1.数据展示 将处理数据送入展示模块
    2.数据训练模块 Dataflow/datautils/Trainer
        常见训练参数：
            epochs
            lr
            batch_size
            num_workers
            gamma

            optimizer
            schedule
            
        保存训练中的最好结果：best_res

        训练函数接口:
            一个epoch中的训练代码：train_func
            一个epoch中的测试代码：val_func
            一个epoch中的结果处理及展示代码:res_func
            训练完成的结束代码:end_func
            若不设置train_func则相当于仅测试

有模型控制模块(model/ModelContorller)
    模型models[]来自各模型.main
    模型points(仅3d点云用)[]来自各模型.point
    数据预处理函数接口predeal[]来自各模型.predeal
    结果处理函数接口resdeal[]来自各模型.resdeal
    损失函数接口lossfuc[]来自各模型.lossfuc
    checkpoint存在在 datasets/datasetp/模型架构名/checkpoints/best.pth 中

有数据展示模块(visualizer/OpencvVis)