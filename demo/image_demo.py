from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    # show_result_pyplot(args.img, result, label, score_thr=args.score_thr)

    import pickle
    import numpy as np
    import tensorflow as tf
    import cv2
    import copy

    label_mapping = pickle.load(open("label_mapping.pkl", "rb"))

    ### LOAD MODEL ###
    model = tf.keras.models.load_model("./milkcan.hdf5")

    img = cv2.imread(args.img)
    dest_img = copy.deepcopy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for bboxes in result:
        for x1, y1, x2, y2, score in bboxes:
            box_img = img[int(y1):int(y2), int(x1):int(x2)]
            resized_box_img = cv2.resize(box_img, (32, 32))
            label_idx = np.argmax(model.predict(array_of_images), axis=-1).tolist()[0]
            label = label_mapping[label_idx]

            dest_img = cv2.rectangle(dest_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            dest_img = cv2.putText(dest_img, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite("result.png", dest_img)

if __name__ == '__main__':
    main()
