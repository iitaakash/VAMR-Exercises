import numpy as np




def Sift(image, num_scales, num_octaves, sigma, contrast_thresh):

    blur_data = []
    dog_data = []
    for j in range(num_octaves):
        X,Y = img.shape[1]*(1.0/(j+1.0)), img.shape[0]*(1.0/(j+1.0))
        image_oct = cv2.resize(img,(int(X),int(Y)))
        scale_sigma = sigma
        data.append([])
        last_image = None
        for k in range(-1, num_scales + 2):
            scale_sigma = 2**((k+1.0)/sigma) * sigma
            image_scale = cv2.GaussianBlur(image_oct, ksize = (5,5), sigmaX = scale_sigma, sigmaY = scale_sigma)
            if k > -1:
                data[-1].append(image_scale - last_image)
            last_image = image_scale

    kpts = []          
    for m,dta in enumerate(data):
        dat = np.array(dta)
        width = dat.shape[1]
        height = dat.shape[2]
        dat[dat < contrast_threshold] = 0.0
        print(dat.shape)
        for n in range(1,num_scales+1):
            for w in range(1,width-1):
                for h in range(1,height-1):
                    dat33 = dat[n,w-1:w+2,h-1:h+2]
                    if IsCenterMax(dat33):
                        kpts.append((w,h,n))

    print(len(kpts))