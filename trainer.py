import torch.nn.init
from Loss import *
from Model import *

def Dimension_Reduction(data_image, args):
    print('Step 1: Dimensionality Reduction ...')
    m, n = data_image.shape

    if args.use_umap:
        data_umap = umap.UMAP(metric = 'cosine',n_components=3,random_state = 0).fit_transform(data_image)
        data_umap = MinMaxScaler().fit_transform(data_umap)
        data_umap = torch.from_numpy(data_umap.astype(np.float32))
        loss_func2 = torch.nn.MSELoss()

        if args.use_gpu == True:
            data_umap = data_umap.cuda()

    data_image = torch.from_numpy(data_image.astype(np.float32))

    if args.use_gpu == True:
        data_image = data_image.cuda()

    model = Autoencoder(n)

    # optimizer = torch.optim.SGD(autoencoder.parameters(), lr=args.lr_dr,momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_dr)
    loss_func1 = CosineLoss()

    for epoch in range(args.epoch_dr):

        if args.use_gpu == True:
            model = model.cuda()

        encoded, deconder = model(data_image)

        if args.use_umap == True:

            loss = loss_func2(encoded, data_umap) + loss_func1(deconder,data_image)

        else:

            loss = loss_func1(deconder,data_image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch', epoch, '| train loss:  %.4f' % loss.data.cpu().numpy())

    torch.save(model.state_dict(), 'FEmodule_weight.pth')

    if args.use_gpu == True:
        torch.cuda.empty_cache()


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_uniform(m.weight, mode='fan_in')
        # nn.init.constant_(m.bias, 0)

def Feature_Clustering(image, args):
    print('Step 2: Feature Clustering ...')
    m, n, k = args.input_shape

    image = image.reshape(m, n, k)

    im = torch.from_numpy(np.array([image.transpose((2, 0, 1)).astype('float32')]))
    if args.use_gpu == True:
        im = im.cuda()

    label_colours = np.random.randint(255, size=(args.nChannel, 3))
    if args.mode == 'spat-config':
        model = FCNetwork_SPAT_spec(k, args)
        model2 = FCNetwork_SPAT_spec(k, args)
        modelAverage = FCNetwork_SPAT_spec(k, args)
        model2Average = FCNetwork_SPAT_spec(k, args)

    if args.mode == 'spat-spor':
        model = FCNetwork_spat_SPEC(k, args)
        model2 = FCNetwork_spat_SPEC(k, args)
        modelAverage = FCNetwork_spat_SPEC(k, args)
        model2Average = FCNetwork_spat_SPEC(k, args)

    model.train()
    model.apply(weight_init)

    model2.train()
    model2.apply(weight_init)

    pretrain = torch.load('FEmodule_weight.pth')

    model_dict = model.state_dict()
    pretrain = {k: v for k, v in pretrain.items() if k in model_dict.keys()}
    model_dict.update(pretrain)

    model_dict2 = model2.state_dict()
    pretrain = {k: v for k, v in pretrain.items() if k in model_dict2.keys()}
    model_dict2.update(pretrain)

    model.load_state_dict(model_dict)
    model2.load_state_dict(model_dict2)

    HPy_target = torch.zeros(image.shape[0] - 1, image.shape[1], args.nChannel)
    HPz_target = torch.zeros(image.shape[0], image.shape[1] - 1, args.nChannel)

    if args.use_gpu == True:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_soft = TripletLoss()
    loss_fn2 = torch.nn.CrossEntropyLoss()

    loss_hpy = torch.nn.L1Loss(size_average=True)
    loss_hpz = torch.nn.L1Loss(size_average=True)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_fc, momentum=args.momentum_fc)
    optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, model2.parameters()), lr=args.lr_fc, momentum=args.momentum_fc)

    # model.parameter = modelAverage.parameter
    for modelPara, modelAvePara in zip(model.parameters(), modelAverage.parameters()):
        modelAvePara.data = modelPara.data
    for modelPara, modelAvePara in zip(model2.parameters(), model2Average.parameters()):
        modelAvePara.data = modelPara.data

    a = []
    for batch_idx in range(args.epoch_fc):
        # forwarding
        # initial parameter and get four output result

        optimizer.zero_grad()
        optimizer2.zero_grad()
        if args.use_gpu == True:
            model.cuda()
            model2.cuda()
            modelAverage.cuda()
            model2Average.cuda()


        outputq, _,xxxx1 = model(im, m, n, k)
        output2q, _,xxxx2 = model2(im, m, n, k)
        outputAverageq, _, xxxx3 = modelAverage(im, m, n, k)
        output2Averageq, xxx,xxxx4 = model2Average(im, m, n, k)

        # deal with ouput
        output = outputq.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
        output2 = output2q.permute(1, 2, 0).contiguous().view(-1, args.nChannel)

        outputAverage = outputAverageq.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
        ignore, Averagetarget = torch.max(outputAverage, 1)
        im_Averagetarget = Averagetarget.data.cpu().numpy()

        output2Average = output2Averageq.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
        ignore, Average2target = torch.max(output2Average, 1)
        im_Average2target = Average2target.data.cpu().numpy()
        ari = adjusted_rand_score(im_Average2target,im_Averagetarget)

        print('ARI is : '+ str(ari))

        # initail tv loss
        outputHP = output.reshape((image.shape[0], image.shape[1] ,  args.nChannel))
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = loss_hpy(HPy, HPy_target)
        lhpz = loss_hpz(HPz, HPz_target)

        outputHP2 = output2.reshape((image.shape[0] , image.shape[1] , args.nChannel))
        HPy2 = outputHP2[1:, :, :] - outputHP2[0:-1, :, :]
        HPz2 = outputHP2[:, 1:, :] - outputHP2[:, 0:-1, :]
        lhpy2 = loss_hpy(HPy2, HPy_target)
        lhpz2 = loss_hpz(HPz2, HPz_target)

        ignore, target = torch.max(output, 1)
        target_latest = target

        ignore, target2 = torch.max(output2, 1)
        target2_latest = target2

        nLabels = len(np.unique(im_Averagetarget))

        if batch_idx % 3 == 0:

            im_target2_rgb = np.array([label_colours[c % args.nChannel] for c in im_Averagetarget])
            im_target2_rgb = im_target2_rgb.reshape(m, n, 3).astype(np.uint8)

            im_target_rgb22 = np.array([label_colours[c % args.nChannel] for c in im_Average2target])
            im_target_rgb22 = im_target_rgb22.reshape(m, n, 3).astype(np.uint8)

            cv2.imwrite(args.output_file + ".jpg", im_target2_rgb)

        if args.mode == 'spat-SPEC':
            args.stepsize_tv = 0

        loss1 = args.stepsize_tv * (lhpy + lhpz) + args.stepsize_sta * loss_soft(output, Average2target,batch_idx) \
                + args.stepsize_sim*loss_fn(output, target_latest)

        loss2 = args.stepsize_tv * (lhpy2 + lhpz2) + args.stepsize_sta * loss_soft(output2, Averagetarget,batch_idx) \
                + args.stepsize_sim*loss_fn(output2, target2_latest)

        loss = loss1 + loss2

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer2.step()

        for modelPara, modelAvePara in zip(model.parameters(), modelAverage.parameters()):
            modelAvePara.data = args.meanNetStep * modelPara.data + (1 - args.meanNetStep) * modelAvePara.data

        for modelPara, modelAvePara in zip(model2.parameters(), model2Average.parameters()):
            modelAvePara.data = args.meanNetStep * modelPara.data + (1 - args.meanNetStep) * modelAvePara.data

        print(batch_idx, '/', args.epoch_fc, ':', nLabels, loss2.item())

        if nLabels <= 2:
            break

    return im_Average2target
