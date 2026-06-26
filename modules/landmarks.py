"""
modules/landmarks.py
---------------------
Static landmark-index groups for the standard MediaPipe 468/478-point
face mesh topology. Pure data — no drawing, no detection logic — so any
module (filters, compositor, pose) can import just the indices it needs.
"""

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]

LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 285, 295]

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159,
            160, 161, 246, 33]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
             386, 385, 384, 398, 362]

LEFT_IRIS = [468, 469, 470, 471, 472, 468]
RIGHT_IRIS = [473, 474, 475, 476, 477, 473]

LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324,
              318, 402, 317, 14, 87, 178, 88, 95, 78, 61]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310,
              311, 312, 13, 82, 81, 80, 191, 78]

NOSE_BRIDGE = [6, 197, 195, 5, 4, 1, 19, 94, 2]

# Single-point references used for geometry/pose calculations
LEFT_EYE_OUTER_CORNER = 33
LEFT_EYE_INNER_CORNER = 133
RIGHT_EYE_INNER_CORNER = 362
RIGHT_EYE_OUTER_CORNER = 263
LEFT_TEMPLE = 127
RIGHT_TEMPLE = 356

# Anchor pool the sparkle effect picks particles from (forehead, nose,
# cheeks — matches the look of the reference AR demo).
SPARKLE_ANCHOR_POOL = [10, 109, 67, 103, 54, 21, 251, 284, 332,
                        197, 195, 5, 4, 1, 19, 94,
                        50, 280, 234, 454, 132, 361, 207, 427, 195, 5]