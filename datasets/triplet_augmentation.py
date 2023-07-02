import albumentations as A


GEOMETRIC_DOUBLE = 'GEOMETRIC_DOUBLE'
GEOMETRIC_SINGLE = 'GEOMETRIC_SINGLE'
FINE_SINGLE = 'FINE_SINGLE'
COLOR_DOUBLE = 'COLOR_DOUBLE'
COLOR_SINGLE = 'COLOR_SINGLE'
RANDOM_CROP_SINGLE = 'RANDOM_CROP_SINGLE'
RANDOM_CROP_DOUBLE = 'RANDOM_CROP_DOUBLE'


def make_train_aug(size=(512, 512)):
    h, w = size
    geometric_aug = [
        A.Flip(p=0.75),
        A.Transpose(p=0.5),
        A.RandomRotate90(p=0.75),
        A.ShiftScaleRotate(scale_limit=(-0.5, 0.5), shift_limit=0, rotate_limit=45, p=0.9),
        A.Perspective(p=0.2),
        A.PadIfNeeded(min_height=h, min_width=w, always_apply=True, border_mode=0),
        ]


    geometric_double= A.Compose(geometric_aug, additional_targets={'positive': 'image'})
    geometric_single = A.Compose(geometric_aug)

    fine_aug = [
        A.Rotate(limit=2, interpolation=1, p=0.5),
        A.RandomSunFlare(p=0.05),
        A.RandomFog(p=0.05),
        A.ElasticTransform(p=0.25),
    ]

    fine_single = A.Compose(fine_aug)

    color_aug = [
        A.Sharpen (alpha=(0.05, 0.1), lightness=(0.1, 0.5), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=1.0),
        A.RGBShift(),
        A.Cutout(p=0.2),
        A.CLAHE(p=0.2),
        A.RandomGamma(p=1),
        A.HueSaturationValue(p=1),
        A.ChannelShuffle(p=0.2),

        A.OneOf([
            A.GaussNoise(p=1),  
            A.Emboss(p=1),
            A.Sharpen(p=1),
            A.JpegCompression(p=1),
        ], p=1),

        A.OneOf([
            A.Blur(blur_limit=3, p=1),
            A.GaussianBlur(blur_limit=3, p=1),
            A.MedianBlur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),    
        ], p=1),
        ]

    color_double = A.Compose(color_aug, additional_targets={'negative': 'image'})
    color_single = A.Compose(color_aug)

    random_crop_double = A.Compose([A.RandomCrop(height=h, width=w, always_apply=True),],
                                   additional_targets={'positive': 'image'})
    random_crop_single = A.Compose([A.RandomCrop(height=h, width=w, always_apply=True),])

    return {
        GEOMETRIC_DOUBLE: geometric_double,
        GEOMETRIC_SINGLE: geometric_single,
        FINE_SINGLE: fine_single,
        COLOR_DOUBLE: color_double,
        COLOR_SINGLE: color_single,
        RANDOM_CROP_SINGLE: random_crop_double,
        RANDOM_CROP_DOUBLE: random_crop_single,
        }


def make_no_aug(size=(512, 512)):
    h, w = size
    geometric_aug = [
        A.PadIfNeeded(min_height=h, min_width=w, always_apply=True, border_mode=0),
        ]


    geometric_double= A.Compose(geometric_aug, additional_targets={'positive': 'image'})
    geometric_single = A.Compose(geometric_aug)

    fine_aug = [
    ]

    fine_single = A.Compose(fine_aug)

    color_aug = [
        ]

    color_double = A.Compose(color_aug, additional_targets={'negative': 'image'})
    color_single = A.Compose(color_aug)

    random_crop_double = A.Compose([A.RandomCrop(height=h, width=w, always_apply=True),],
                                   additional_targets={'positive': 'image'})
    random_crop_single = A.Compose([A.RandomCrop(height=h, width=w, always_apply=True),])

    return {
        GEOMETRIC_DOUBLE: geometric_double,
        GEOMETRIC_SINGLE: geometric_single,
        FINE_SINGLE: fine_single,
        COLOR_DOUBLE: color_double,
        COLOR_SINGLE: color_single,
        RANDOM_CROP_SINGLE: random_crop_double,
        RANDOM_CROP_DOUBLE: random_crop_single,
        }