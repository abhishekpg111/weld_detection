import numpy as np
import cv2 
import image_geometry
import argparse
import time
from skimage import morphology
import skimage
import datetime
start_time = time.time()


global image_no
image_no=1

def show_image(window_name,image):
	global image_no
	cv2.namedWindow(window_name)        # Create a named window
	cv2.moveWindow(window_name, 40,30)  # Move it to (40,30)
	cv2.imshow(window_name,image)
	cv2.waitKey()
	cv2.destroyAllWindows()	
	name="output/"+str(image_no)+"_"+str(window_name)+"_"+str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')+".png"
	#print name
	image_no+=1
	cv2.imwrite(name,image)



def camera_input(show=1):

	""" function to read the image from user"""
	foreground=raw_input("Enter the foreground name : ")
	background=raw_input("Enter the background image name : ")
	rgb=cv2.imread(foreground)
	rgb=cv2.resize(rgb,(640,480))
	depth=cv2.imread(background)
	depth=cv2.resize(depth,(640,480))
	if show:
		show_image("foreground",rgb)
		show_image("background",depth)
	return rgb,depth

def auto_canny(image,show=0,sigma=1):
	

	# compute the median of the single channel pixel intensities
	img = image.copy()
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(img, lower, upper)
	if show:
		show_image("canny_image",edged)	 
	# return the edged image
	return edged

def edged(images,show=0):

	#images=cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
	prev_i=0
	img=images.copy()
	for i in range(40, 481,40):
		prev_j=0
		for j in range(40, 641,40):
		#print prev_i,i,prev_j,j
			roi=img[prev_i:i,prev_j:j]
			blur = cv2.GaussianBlur(roi,(5,5),0)
			sobel_temp=auto_canny(blur,)
			#ret2_temp,otsu_temp = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			img[prev_i:i,prev_j:j]=sobel_temp
			prev_j=j
	prev_i=i
	if show:
		show_image("edged",images)
	return img


def skeletonize(img):

    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break
    return skel

def remove_noise(skelton,min_length,show=0):

	imglab = skimage.measure.label(skelton) # create labels in segmented image
	cleaned = morphology.remove_small_objects(imglab, min_size=min_length, connectivity=2)
	clean_image = np.zeros((cleaned.shape)) # create array of size cleaned
	clean_image[cleaned > 0] = 255
	if show:
		show_image("remove_noise",clean_image)
	return clean_image

def filter_contour(rgb,threshold,arc_length,show=0):

	""" filter off small contours with arclength less than 20 pixels """
	
	rgb_copy=rgb.copy()
	image, contours, hierarchy = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
	#print len(contours)
	filtered_contour=[]
	for contour in contours:
		perimeter = cv2.arcLength(contour,False)
		if perimeter>arc_length:
			filtered_contour.append(contour)
	#print len(filtered_contour)
	if show:
		con=cv2.drawContours(rgb_copy, filtered_contour, -1, (0,255,0), 2)
		show_image("filtered contour",con)	
	return filtered_contour


def roi_filter(rgb,skelton,roi,show=0):
	rgb_copy=rgb.copy()
	initial_seed=[]
	for i in range(0,480):
		for j in range(0, 640):
			px=skelton[i,j]
			if px==255 and i>=roi[0] and j>=roi[2] and i<=roi[1] and j<=roi[3]:
				rgb_copy[i,j]=[0,0,255]
				initial_seed.append([i,j])

	if show:
		show_image("Initial_seeds",rgb_copy)
	return initial_seed
		



def initial_linegrow(rgb,initial_seed,seed_no,threshold,show=0):
 
	 rgb_copy=rgb.copy()
	 rgb_copy1=rgb.copy()
	 filtered_seeds=[]
	 for seed in initial_seed:
		  #print seed
		  line_temp=cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
		  prev_d=0
		  sm_th=[0]*8
		  closed_list=[[0]*2]*100 
		  curr_node=[[]]
		  j=0
		  while j<seed_no:
			d=[]
			if j==0:
				curr_node[0]=seed
				#print curr_node
			for i in range(0,8):
				if i==0:
					xoff=1
					yoff=0
				if i==1:
					xoff=1
					yoff=1
				if i==2:
					xoff=0
					yoff=1
				if i==3:
					xoff=-1
					yoff=1
				if i==4:
					xoff=-1
					yoff=0
				if i==5:
					xoff=-1
					yoff=-1
				if i==6:
					xoff=0
					yoff=-1
				if i==7:
					xoff=1
					yoff=-1
				k=1
				#print i
				#print curr_node[0][0]+xoff*k,curr_node[0][1]+yoff*k
				in_sum=0
				while k<=3:
					if [curr_node[0][0]+yoff*k,curr_node[0][1]+xoff*k] in closed_list:
						in_sum=1000
						break
					in_sum=in_sum + line_temp[curr_node[0][0]+yoff*k,curr_node[0][1]+xoff*k]
					k=k+1
				d.append(in_sum)
			#print d
			min_d=d.index(min(d))
	
			#print min_d
			if min_d==0:
				xoff=1
				yoff=0
			if min_d==1:
				xoff=1
				yoff=1
			if min_d==2:
				xoff=0
				yoff=1
			if min_d==3:
				xoff=-1
				yoff=1
			if min_d==4:
				xoff=-1
				yoff=0
			if min_d==5:
				xoff=-1
				yoff=-1
			if min_d==6:
				xoff=0
				yoff=-1
			if min_d==7:
				xoff=1
				yoff=-1	
			closed_list[j]=curr_node[0]
			line_temp[curr_node[0][0]-1:curr_node[0][0]+2,curr_node[0][1]-1:curr_node[0][1]+2]=255 ## setting all the pixels around curent node white (255)
			curr_node=[[curr_node[0][0]+yoff,curr_node[0][1]+xoff]]

			if show==1:

				rgb_copy[closed_list[j][0],closed_list[j][1]]=[0,255,0]
				#show_image("line_growing",rgb_copy)
			if j>0 and prev_d-min_d<>0:
				sm_th[min_d]=1
			prev_d=min_d
			#print sm_th
			j=j+1

		  if np.sum(sm_th)<=threshold :
			filtered_seeds.append(seed)
			if show==1:

				rgb_copy1[seed[0]-2:seed[0]+2,seed[1]-2:seed[1]+2]=[0,0,255] ## uncomment this to show the final seeds having straight lines
	 if show==1:
		show_image("line_growing",rgb_copy)
		show_image("final_seeds",rgb_copy1)

	 return filtered_seeds

def best_seed(rgb,final_seed,show=0):

 
	line_temp=cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
	inten=2550
	for seed in final_seed:
		intencity=np.sum(np.sum(line_temp[seed[0]-1:seed[0]+2,seed[1]-1:seed[1]+2]))
		if intencity<inten:
			inten=intencity
			final_node=seed
			#print seed,intencity
	if show:
		rgb_copy=rgb.copy()
		rgb_copy[final_node[0]-2:final_node[0]+2,final_node[1]-2:final_node[1]+2]=[0,255,0]
		show_image("best_seed",rgb_copy)
	return final_node

def final_grow(rgb,final_seed,threshold,show=0):
	
	rgb_copy=rgb.copy()
	line_temp=cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
	closed_list=[[0]*2]*1000 
	curr_node=[[]]
	j=0
	loop=0
	prev_loop=0
	final_list=[[]]
	for z in (1,3):
	 prev_d=0
	 test_node=[final_seed]
	 sm_th=[0]*8
	 flag=0
	 while flag==0:
	
		#print loop
		d=[]
		if j==0 or j==loop:
			curr_node[0]=test_node[0]
			#print "curr_node",curr_node[0]
			#print curr_node
		for i in range(0,8):
			#print "hi"
			if i==0:
				xoff=1
				yoff=0
			if i==1:
				xoff=1
				yoff=1
			if i==2:
				xoff=0
				yoff=1
			if i==3:
				xoff=-1
				yoff=1
			if i==4:
				xoff=-1
				yoff=0
			if i==5:
				xoff=-1
				yoff=-1
			if i==6:
				xoff=0
				yoff=-1
			if i==7:
				xoff=1
				yoff=-1
			k=1
			in_sum=0
			while k<=3:
				if [curr_node[0][0]+yoff*k,curr_node[0][1]+xoff*k] in closed_list:
					in_sum=1000
					break
				if curr_node[0][0]+yoff*k >= 480 or curr_node[0][1]+xoff*k >= 640:
		   			if z==1:
						print closed_list[j-1::-1]
						final_list.extend(closed_list[j-1::-1])
						prev_loop=j
		   			else:
						print closed_list[prev_loop+1:j]
						final_list.extend(closed_list[prev_loop+1:j])
		   	
		   			loop=j
					flag=1
					break
				else:
					in_sum=in_sum + line_temp[curr_node[0][0]+yoff*k,curr_node[0][1]+xoff*k]
					loop=j
				k=k+1
			d.append(in_sum)
		#print d
		min_d=d.index(min(d))
	
		#print min_d
		if min_d==0:
			xoff=1
			yoff=0
		if min_d==1:
			xoff=1
			yoff=1
		if min_d==2:
			xoff=0
			yoff=1
		if min_d==3:
			xoff=-1
			yoff=1
		if min_d==4:
			xoff=-1
			yoff=0
		if min_d==5:
			xoff=-1
			yoff=-1
		if min_d==6:
			xoff=0
			yoff=-1
		if min_d==7:
			xoff=1
			yoff=-1	
		closed_list[j]=curr_node[0]
		line_temp[curr_node[0][0]-1:curr_node[0][0]+2,curr_node[0][1]-1:curr_node[0][1]+2]=255
		curr_node=[[curr_node[0][0]+yoff,curr_node[0][1]+xoff]]
		if show:
			rgb_copy[closed_list[j][0]-1:closed_list[j][0]+1,closed_list[j][1]-1:closed_list[j][1]+1]=[0,255,0] ## line that showing final weld seem
		#print j,prev_d-min_d
		if j>0 and prev_d-min_d<>0:
			sm_th[min_d]=1
		prev_d=min_d
		#print np.sum(sm_th)
		j=j+1
		if np.sum(sm_th)>threshold:
		   if z==1:
			#print closed_list[j-1::-1]
			final_list.extend(closed_list[j-1::-1])
			prev_loop=j
		   else:
			#print closed_list[prev_loop+1:j]
			final_list.extend(closed_list[prev_loop+1:j])
		   	
		   loop=j
		   flag=1
		   break
	#print final_list
	final_list.pop(0)
	if show:
		show_image("final_weld",rgb_copy)
	return final_list

def trim_weld(rgb,final_weld,seed_no,threshold,show=0):

	rgb_copy=rgb.copy()

	for i in range(len(final_weld)):
		wrong_seeds=[]
		check_trim=initial_linegrow(rgb,[final_weld[i]],seed_no,threshold,0)
		if check_trim!=[]:
			#print i
			break
	final_weld=final_weld[i:]
	for i in range(len(final_weld)):
		check_trim=initial_linegrow(rgb,[final_weld[len(final_weld)-1-i]],seed_no,threshold,0)
		if check_trim!=[]:
			break
	final_weld=final_weld[:len(final_weld)-i]
	if show:
		for [i,j] in final_weld:
			rgb_copy[i-2:i+1,j-2:j+1]=[0,255,0]
		show_image("trimm",rgb_copy)

	return final_weld

if __name__=='__main__':


	rgb,depth=camera_input()
	foreground=cv2.subtract(depth,rgb)
	gray=cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
	canny=auto_canny(gray,0)
	clean_image_canny=remove_noise(canny,20,0)
	kernel = np.ones((5,5),np.uint8)
	dilation = cv2.dilate(clean_image_canny,kernel,iterations = 1)
	erosion = cv2.erode(dilation,kernel,iterations = 1)
	skelton=skeletonize(erosion)
	initial_seed=roi_filter(rgb,skelton,[120,370,120,520],0)
	print "\nNumber of Initial seeds : ",len(initial_seed)
	final_seeds=initial_linegrow(rgb,initial_seed,100,2,0)
	best_sed=best_seed(rgb,final_seeds,0)
	print "\nbest seed : ",best_sed
	final_weld=final_grow(rgb,best_sed,3,0)
	print "\nbefore trim :",len(final_weld)
	final_weld=trim_weld(rgb,final_weld,20,2,1)
	print "\nafter trim :",len(final_weld)
	print("\n--- %s seconds ---" % (time.time() - start_time))
