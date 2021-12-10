#coding: utf-8
import os
import cv2

def get_file_path_list(json_dir, postfix = [".jpg"] ):
    '''  '''
    file_path_list = []
    if os.path.exists(json_dir):
        print (json_dir)
    else:
        print ("Do not exist")
    for rt, dirs, files in os.walk(json_dir):
        if len(dirs)>0:
            continue
        for file in files:
            if os.path.splitext(file)[1] in postfix:
                file_path_list.append( os.path.join(rt, file ) )
    return file_path_list


def extract_frame_from_video(video_path, video_frame_save_dir ):
    '''
    :param video_path:
    :param video_frame_save_dir:
    :return: 
    '''
    Parent_dir, Video_name = os.path.split(video_path )
    VideoName_prefix, VideoName_postfix = os.path.splitext(Video_name)
    VideoNamePath_txt = VideoName_prefix+"_path"+".txt"
    VideoNameResult_txt = VideoName_prefix+"_result"+".txt"
    fwre = open(os.path.join(os.path.split(video_frame_save_dir)[0], VideoNameResult_txt), "w")
    fwre.close()
    fw = open(os.path.join(os.path.split(video_frame_save_dir)[0], VideoNamePath_txt), "w")

    video_object = cv2.VideoCapture(video_path )
    frame_index = 1
    while True:
        ret, frame = video_object.read()
        if ret == False:
            print("extract_frame_from_video(), extract is finished")
            return
        frame_name = VideoName_prefix+ "_%0*d"%(6,frame_index) + ".jpg"
        frame_index += 1
        #cv2.imwrite( os.path.join(video_frame_save_dir, frame_name), frame )
        #video_frame_save_dir = video_frame_save_dir[15:]
        frame_name_path = os.path.join(video_frame_save_dir, frame_name)+" "+str(os.path.splitext(Video_name)[0])+"\n"
        #frame_name_path.replace("/share/caiyuanqiang/VideoSet/NewVideo_Frames","/VideoSet/NewVideo_Frames")
        #print(frame_name_path, len(frame_name_path), "KSText_ReDetRec_20200118/"+frame_name_path[29:])
        fw.write("KSText_ReDetRec_20200118/"+frame_name_path[29:])
    fw.close()

def batch_extractFrame_fromVideo(VideoSet_dir=""):
    '''
    :return: 
    '''
    if VideoSet_dir == "":
        print("Please input video folder")
        return
    VideoDir_list = get_file_path_list(VideoSet_dir, postfix = [".mp4"] )
    VideoDir_list.sort()
    print("KSText==> Extract frames from video")

    for Video_dir in VideoDir_list:
        print(Video_dir)
        Parent_dir, Video_name = os.path.split(Video_dir )
        Video_prefix, Video_postfix = os.path.splitext(Video_name)
        Parent_dir_new = Parent_dir + "_Frames"
        if not os.path.exists(Parent_dir_new):
            os.makedirs(Parent_dir_new)
        NewVideoFrame_dir = os.path.join(Parent_dir_new, Video_prefix )
        if not os.path.exists(NewVideoFrame_dir):
            os.makedirs(NewVideoFrame_dir )
        extract_frame_from_video(Video_dir, NewVideoFrame_dir)


if __name__ == "__main__":
    print ("Hello CAI!")
    VideoSetDir_list = []
    
    VideoSetDir_list.append('/share/caiyuanqiang/VideoSet/NewVideo')
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls10_YL")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls11_YS")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls12_QG")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls13_FCJJ")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls14_CY")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls15_SY")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls16_ZWH")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls17_JY")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls18_LX")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls19_SS")

    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls1_ZYDS")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls20_XY")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls21_MY")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls22_QC")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls23_HW")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls24_YY")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls25_DJ")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls26_KJ")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls27_KP")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls28_MY")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls29_MZ")

    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls2_ECY")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls30_WD")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls31_CJ")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls32_YY")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls3_TY")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls4_MRMX")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls5_PP")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls6_XW")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls7_YX")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls8_XJ")
    #VideoSetDir_list.append("/share/caiyuanqiang/VideoSet/Cls9_XYH")






    for VideoSet_dir in VideoSetDir_list:
        print(VideoSet_dir)
        batch_extractFrame_fromVideo(VideoSet_dir)


