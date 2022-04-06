import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image

Depth_model = tf.keras.models.load_model(
    'D:/TFODCourse-main/TFODCourse-main/Tensorflow/workspace/models/my_ssd_mobnet/export/saved_model/saved_model.pb')

class_names = ['person']


vdo_path = 'D:/Project/vdo/test/sit_jar/'
vdo_dir_name = 'newvdo'
vdo_dir_name = str(input("Enter VDO_DIR_NAME <= "))
vdo_path += vdo_dir_name + '/'

# depth_pathwithout ="D:\รวมงานทั้งหมด\final-project\src\clone\image\ท่าปกติ\นั่งบนโถ\image_depth\without_background\\"
# depth_pathwit ="D:\รวมงานทั้งหมด\final-project\src\clone\image\ท่าปกติ\นั่งบนโถ\image_depth\with_background\\"

# ir_pathwithout="D:\รวมงานทั้งหมด\final-project\src\clone\image\ท่าปกติ\นั่งบนโถ\image_ir\without_background"
# ir_pathwit="D:\รวมงานทั้งหมด\final-project\src\clone\image\ท่าปกติ\นั่งบนโถ\image_ir\with_background"

depth_stream = cv2.VideoCapture(vdo_path + "depth.avi")
ir_stream = cv2.VideoCapture(vdo_path + "ir.avi")

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=None, varThreshold=None, detectShadows=True)
# fgbg.setHistory(600)
# fgbg.setNMixtures(5)
fgbg.setDetectShadows(False)
fgbg.setVarThreshold(95)


while True:
    ret, frame_depth = depth_stream.read()
    ret, frame_ir = ir_stream.read()

    fgmask_show = None

    # Decode Depth
    # 8upperbits data in ch1 / 8lowerbits data in ch2 / ignore ch3
    depth_ch1, depth_ch2, _ = cv2.split(frame_depth)
    # 8upperbits data -> convert uint16 and shift left << 8
    decoded_depth = np.left_shift(np.uint16(depth_ch1.copy()), 8)
    decoded_depth = np.bitwise_or(decoded_depth, np.uint16(
        depth_ch2.copy()))  # bitwise or with 8lowerbits

    # Decode IR
    # 8upperbits data in ch1 / 8lowerbits data in ch2 / ignore ch3
    ir_ch1, ir_ch2, _ = cv2.split(frame_ir)
    # 8upperbits data -> convert uint16 and shift left << 8
    decoded_ir = np.left_shift(np.uint16(ir_ch1.copy()), 8)
    decoded_ir = np.bitwise_or(decoded_ir, np.uint16(
        ir_ch2.copy()))  # bitwise or with 8lowerbits

    # Adjusted for display
    decoded_depth *= 10
    decoded_ir *= 400

    # Get frame id
    frameID = depth_stream.get(cv2.CAP_PROP_POS_FRAMES)

    # Learning background
    if frameID <= 10:
        fgmask = fgbg.apply(decoded_depth, learningRate=0.8)

    elif frameID <= 100:
        fgmask = fgbg.apply(decoded_depth, learningRate=0.5)

    else:
        fgmask = fgbg.apply(decoded_depth, learningRate=0)

    # Erode
    rect3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # cross3x3 = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # ellipse3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    rect5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # cross5x5 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    # ellipse5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    rect9x9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # Mop
    # anti_noise = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, rect5x5, iterations=1)

    # eroded_rect5x5_img = cv2.erode(fgmask, rect5x5, iterations = 1)
    # eroded_cross5x5_img = cv.erode(bin_img, cross5x5, iterations = 1)
    # eroded_ellipse5x5_img = cv.erode(bin_img, ellipse5x5, iterations = 1)
    anti_noise = cv2.erode(fgmask, rect9x9, iterations=1)

    # Dilate
    anti_noise = cv2.dilate(anti_noise, rect9x9, iterations=1)

    # Contour
    # canny = cv2.Canny(anti_noise, 30, 100)
    lapas = cv2.Laplacian(anti_noise, cv2.CV_16S, ksize=3)
    lapas = np.uint8(lapas / 257)
    # print(lapas.dtype)
    contours, hierarchy = cv2.findContours(
        lapas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawn_contour = np.zeros(lapas.shape, dtype=np.uint8)
    drawn_contour = cv2.cvtColor(drawn_contour, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(drawn_contour, contours, -1, (0, 255, 200), 2)

    max_area = 0
    max_contourID = -1
    # Find bounary
    for index, val in enumerate(contours):
        x, y, w, h = cv2.boundingRect(val)
        area = w * h

        if area > max_area and area > 10000:
            max_area = area
            max_contourID = index

    if max_contourID != -1:
        x, y, w, h = cv2.boundingRect(contours[max_contourID])
        cv2.rectangle(anti_noise, (x, y), (x+w, y+h),
                      (255, 255, 255), thickness=2)
        # print(f"w*h = {w*h}")
        debug_contour = np.zeros(lapas.shape, dtype=np.uint8)
        debug_contour = cv2.cvtColor(debug_contour, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(debug_contour, contours,
                         max_contourID, (0, 255, 200), 2)

        # Fill
        bg_fill = np.zeros(fgmask.shape, dtype=np.uint8)
        bg_fill = cv2.fillPoly(
            bg_fill, pts=[contours[max_contourID]], color=(255, 255, 255))

        # Crop
        anti_noise_crop = bg_fill[y:y+h, x:x+w]
        deDepth_person = decoded_depth[y:y+h, x:x+w]
        _, personBin = cv2.threshold(
            anti_noise_crop, 200, 255, cv2.THRESH_BINARY)
        personBin_16bits = np.uint16(personBin) * 257
        person = np.bitwise_and(
            personBin_16bits, deDepth_person)  # 16bits = 65_535
        person_8bits = np.uint8(person / 257)
        deDepth_person8bits = np.uint8(deDepth_person / 257)

        # About IR
        deIR_person = decoded_ir[y:y+h, x:x+w]
        IRperson = np.bitwise_and(personBin_16bits, deIR_person)
        IRperson_8bits = np.uint8(IRperson / 257)
        deIR_person8bits = np.uint8(deIR_person / 257)

        # BgFill
        # bg_fill_crop = bg_fill[y:y+h, x:x+w]
        # BGFill_deDepth_person = decoded_depth[y:y+h, x:x+w]
        # _, BgFill_personBin = cv2.threshold(bg_fill_crop, 200, 255, cv2.THRESH_BINARY)
        # BgFill_personBin_16bits = np.uint16(BgFill_personBin) * 257
        # BgFill_person = np.bitwise_and(BgFill_personBin_16bits, BGFill_deDepth_person) # 16bits = 65_535
        # BgFill_person_8bits = np.uint8(BgFill_person / 257)
        # deDepth_person8bits = np.uint8(deDepth_person / 257)

        # Im write --------------------------------------------------------
        # Depth

        # cv2.imwrite('./image/Depth_NoBg/test/walk/' + vdo_dir_name +
        #             str(int(frameID)) + '.png', person_8bits)   # Write images
        # cv2.imwrite('./image/Depth_Bg/test/walk/' + vdo_dir_name +
        #             str(int(frameID)) + '.png', deDepth_person8bits)  # With BG
        # # IR
        # cv2.imwrite('./image/Ir_NoBg/test/walk/' + vdo_dir_name +
        #             str(int(frameID)) + '.png', IRperson_8bits)
        # cv2.imwrite('./image/Ir_Bg/test/walk/' + vdo_dir_name +
        #             str(int(frameID)) + '.png', deIR_person8bits)

        # -----------------------------------------------------------------
        # resizedper_8bit = cv2.resize(person_8bits, (180, 180))
        # resizedper_8bit = cv2.cvtColor(resizedper_8bit, cv2.COLOR_GRAY2RGB)
        # resizedper_8bit = np.float32(resizedper_8bit)
        # print(resizedper_8bit.shape)
        # resultDepth = Depth_model.predict([resizedper_8bit])
# ----------------------------------------------------------------------------------------------------------------------
        person_8bits = cv2.cvtColor(person_8bits, cv2.COLOR_GRAY2RGB)
        PilDepth = Image.fromarray(person_8bits, 'RGB')
        PilDepth = PilDepth.resize((320, 320))
        PilDepth = np.array(PilDepth)/255.0
        PilDepth = np.expand_dims(PilDepth, axis=0)
        resultDepth = Depth_model.predict(PilDepth)
        # print(resultDepth[0])

        # score = tf.nn.softmax(resultDepth)
        score = resultDepth[0]
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))

        )
        # print(score*100)
# ----------------------------------------------------------------------------------------------------------------------
        # IRperson_8bits = cv2.cvtColor(IRperson_8bits, cv2.COLOR_GRAY2RGB)
        # PilDepth = Image.fromarray(IRperson_8bits, 'RGB')
        # PilDepth = PilDepth.resize((224, 224))
        # PilDepth = np.array(PilDepth)
        # PilDepth = np.expand_dims(PilDepth, axis=0)
        # resultDepth = Depth_model.predict(PilDepth)

        # score = tf.nn.softmax(resultDepth[0])
        # print(
        #     "This image most likely belongs to {} with a {:.2f} percent confidence."
        #     .format(class_names[np.argmax(score)], 100 * np.max(score))
        # )
        fgmask_show = fgmask.copy()
        fgmask_show = cv2.cvtColor(fgmask_show, cv2.COLOR_GRAY2BGR)
        cv2.putText(fgmask_show, "This is {} with a {:.2f}%".format(class_names[np.argmax(score)], 100 * np.max(score)),
                    org=(70, 70),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1.0,
                    color=(0, 0, 255),
                    thickness=3)

        # Imshow ------------------------------------------------------------------------------------------
        # cv2.imshow('Person', person)
        cv2.imshow('Depth Person', deDepth_person8bits)  # ความลึกมีพื้นหลัง
        # cv2.imshow('Anti-noise Crop', anti_noise_crop)
        # cv2.imshow('Debug_contour', debug_contour)
        # cv2.imshow('BgFill', bg_fill)
        cv2.imshow('Person_8bits', person_8bits)  # ความลึกไม่มีพื้นหลัง
        cv2.imshow('IRperson_8bits', IRperson_8bits)  # ir ไม่มีพื้นหลัง
        # cv2.imshow('BgFill_8bits', BgFill_person_8bits)

    # cv2.imshow('Original', fgmask)
    if fgmask_show is not None:
        cv2.imshow('FG Mask', fgmask_show)
    else:
        cv2.imshow('FG Mask', fgmask)
    # cv2.imshow('Original', fgmask_show)

    # cv2.imshow('Anti-noise', anti_noise)
    # cv2.imshow('Lapas', lapas)
    # cv2.imshow('drawn_contour', drawn_contour)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # if frameID == 124:
    #     cv2.waitKey()

depth_stream.release()
ir_stream.release
cv2.destroyAllWindows()
