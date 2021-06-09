import imageio
import matplotlib
import matplotlib.pyplot as plt
import os
%matplotlib inline

#find dimensions of image
def picDimensions(img_folder):
    imHeights = []
    imWidths = []
    for dir in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder,dir)):
            image = imageio.imread(os.path.join(img_folder, dir, file))
            imHeights.append(image.shape[0])
            imWidths.append(image.shape[1])
    
    return imHeights, imWidths
    
    #find pictures with the smallest height and width
def findSmallest(img_folder):
    smallestHeight = 1000000
    smallestHeightClass = ""
    smallestHeightName = ""
    smallestWidth = 1000000
    smallestWidthClass = "" 
    smallestWidthName = "" 
    for dir in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder,dir)):
            image = imageio.imread(os.path.join(img_folder, dir, file))
            imHeight = image.shape[0]
            imWidth = image.shape[1]
            if imHeight <= smallestHeight:
                smallestHeight = imHeight
                smallestHeightClass = dir
                smallestHeightName = file
            if imWidth <= smallestWidth:
                smallestWidth = imWidth
                smallestWidthClass = dir
                smallestWidthName = file
    
    print('smallest Height : ' , smallestHeight, smallestHeightClass, smallestHeightName)
    print('smallest Width : ' , smallestWidth, smallestWidthClass, smallestWidthName)

#display plots of image dimensions    
Heights, Widths = picDimensions(r'GA_Birds/train')
fig1, ax1 = plt.subplots()
ax1.set_title('Heights')
ax1.boxplot(Heights)
fig2, ax2 = plt.subplots()
ax2.set_title('Widths')
ax2.boxplot(Widths)
