# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import glob
import cv2
import shutil
import traceback
import numpy as np


def tif2jpg(images_dir, output_dir):
  image_files = glob.glob(images_dir + "/*.tif")
  for image_file in image_files:
    basename = os.path.basename(image_file)
    image = cv2.imread(image_file)
    basename = basename.replace(".tif", ".jpg")
    output_file = os.path.join(output_dir, basename)
    cv2.imwrite(output_file, image)
    print("--- Saved {}".format(output_file))


if __name__ == "__main__":
  try:
     images_dir = "./mini_test/masks_tif"
     output_dir = "./mini_test/masks"
     if os.path.exists(output_dir):
       shutil.rmtree(output_dir)
     os.makedirs(output_dir)
     tif2jpg(images_dir, output_dir)

  except:
    traceback.print_exc()
 
