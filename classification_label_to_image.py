def classification_label2_image(testing_result_class1,filename):
    import numpy as np
    import cv2
    import scipy.io as sio
    import matplotlib.pyplot as plt
    r=[];g=[];b=[];
    prow=testing_result_class1.shape[0]
    pcolm=testing_result_class1.shape[1]
    im3=np.zeros([prow,pcolm,3],np.uint8)
    im2=np.zeros([prow,pcolm,3],np.uint8)
    for row in range (1,prow):
        for colm in range(1,pcolm):
            i=testing_result_class1[row,colm];
            #red value
            if i==0:
                r=0;
                g=0;
                b=0;
                # green value
            elif i==6:
                r=255;
                g=0;
                b=0;
                # green value
            elif i==3:
                r=0;
                g=255;
                b=0;
                #cyan value
            elif i==1:
                r=0;
                g=0;
                b=255;

            #yellow value
            elif i==4:
                r=255;
                g=255;
                b=0;
           #magenta value
            elif i==5:
                r=255;
                g=0;
                b=255;
            #Cyan value
            elif i==2:
                r=0;
                g=255;
                b=255;
            #white
            elif i==7:
                 r=255;
                 g=255;
                 b=255;
            #organge
            elif i==8:
                 r=255;
                 g=165;
                 b=0;
            #Navy Blue
            elif i==9:
                 r=0;
                 g=0;
                 b=128;
            #Aqua 
            elif i==10:
                 r=0;
                 g=128;
                 b=128;
            #medium gray
            elif i==11:
                 r=128;
                 g=128;
                 b=128;				 
            # RGB
            im3[row,colm,0]=np.uint8(r);
            im3[row,colm,1]=np.uint8(g);
            im3[row,colm,2]=np.uint8(b);
            # BGR
            im2[row,colm,0]=np.uint8(b);
            im2[row,colm,1]=np.uint8(g);
            im2[row,colm,2]=np.uint8(r);


    plt.imshow(im3)
    plt.axis('off')
    cv2.imwrite(filename+'_cv2.png',im2)
    plt.imsave(filename+"_plt.png",im3)
    #plt.show()

    return im3
