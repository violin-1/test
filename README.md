有数据控制模块(Dataflow/Datacontroller)
    该模块中:
    外置初始化接口，Set_datactr_func
    执行模块接口，exec_func
    加载数据类型，datatsetstype []
    数据显示参数设置接口，datasets_vis_arg_func []
    数据操作设备:device
    内嵌模型控制

常见的数据操作分为：
    1.数据展示： 将处理数据送入展示模块
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
        该模块提供自定义初始化接口:train_init

        训练函数接口:
            一个epoch中的训练代码：train_func
            一个epoch中的测试代码：val_func
            一个epoch中的结果处理及展示代码:res_func
            训练完成的结束代码:end_func
            若不设置train_func则相当于仅测试
        logger模块:
            训练模型类型：modeltype
            训练数据集类型：datasettype
            数据内参：dataarch

            logger可以根据自己需求调整__init__()

            logger提供保存路径：
            断点保存路径:savedir
            checkpoints 默认保存在 savedir
            日志保存路径:logdir
            log默认保存在logdir中
有模型控制模块(model/ModelContorller)
    模型models[]来自各模型.main
    模型points(仅3d点云用)[]来自各模型.point
    数据预处理函数接口predeal[]来自各模型.predeal
    结果处理函数接口resdeal[]来自各模型.resdeal
    损失函数接口lossfuc[]来自各模型.lossfuc
    checkpoints默认保存在 datasets/datasetp/模型架构名/checkpoints/best.pth 中
    logs默认保存在datasets/datasetp/模型架构名/logs 中

有数据展示模块(visualizer/OpencvVis)
    提供2d，3d展示功能：
    点云数据标志:_3d默认为True(False则为2D数据)

    数据获取接口:grab_pcl_fuc
    自定义参数设置接口:arg_set_func
    (
        自定义模型切换接口:switch_model_func
        自定义数据集切换接口:switch_dataset_func
        自定义数据项切换接口:switch_pcl_func
    )
    数据主体函数:run_pcl_dsiplay_process
        数据获取，数据渲染，用户交互
    数据待渲染3D点云数据:xyz(3D数据传入接口)
    数据渲染后展示数据:show(2D数据传入接口)

    3D数据：
        三轴数据反转参数：xyzrelist默认为[1,-1,1]
        点云渲染精度:ballradius默认为2
        点云调色板:cmap默认为plt.cm.cool
        渲染后点云的背景:background默认为(0,0,0)
        渲染窗口高:cvh默认为128*6
        渲染窗口宽:cvw默认为128*9
        渲染深度范围:cvz默认为300*3
        是否冻结旋转:freezerot默认为False
        是否对颜色进行归一化处理:normalizecolor默认为True
        渲染间隔:waittime默认为0
    