explicid_isic_dict = {
    'color': ['highly variable, often with multiple colors (black, brown, red, white, blue)',   'uniformly tan, brown, or black',  'translucent, pearly white, sometimes with blue, brown, or black areas',   'red, pink, or brown, often with a scale', 'light brown to black',   'pink brown or red', 'red, purple, or blue'],
    'shape': ['irregular', 'round', 'round to irregular', 'variable'],
    'border': ['often blurry and irregular', 'sharp and well-defined', 'rolled edges, often indistinct'],
    'dermoscopic patterns': ['atypical pigment network, irregular streaks, blue-whitish veil, irregular',  'regular pigment network, symmetric dots and globules',  'arborizing vessels, leaf-like areas, blue-gray avoid nests',  'strawberry pattern, glomerular vessels, scale',   'cerebriform pattern, milia-like cysts, comedo-like openings',    'central white patch, peripheral pigment network', 'depends on type (e.g., cherry angiomas have red lacunae; spider angiomas have a central red dot with radiating legs'],
    'texture': ['a raised or ulcerated surface', 'smooth', 'smooth, possibly with telangiectasias', 'rough, scaly', 'warty or greasy surface', 'firm, may dimple when pinched'],
    'symmetry': ['asymmetrical', 'symmetrical', 'can be symmetrical or asymmetrical depending on type'],
    'elevation': ['flat to raised', 'raised with possible central ulceration', 'slightly raised', 'slightly raised maybe thick']

}

explicid_isic_binary_dict = {
    'asymmetry': ['symmetrical', 'asymmetrical'],
    'border': ['regular', 'abnormal'],
    'color': ['uniform', 'miscellaneous'],
    'surface texture': ['smooth', 'rough/ulcerated'],
    'vascular patterns': ['failed to find', 'there are visible '],
    'elevation': ['flat', 'raised']

}

explicid_isic_minimal_dict = {
    'color': ['typical for Actinic Keratoses', 'typical for Basal Cell Carcinoma', 'typical for Benign Keratosis-like Lesions', 'typical for Dermatofibroma', 'typical for Melanoma', 'typical for Melanocytic Nevus', 'typical for Vascular Lesions'],
    'shape': ['typical for Actinic Keratoses', 'typical for Basal Cell Carcinoma', 'typical for Benign Keratosis-like Lesions', 'typical for Dermatofibroma', 'typical for Melanoma', 'typical for Melanocytic Nevus', 'typical for Vascular Lesions'],
    'border': ['typical for Actinic Keratoses', 'typical for Basal Cell Carcinoma', 'typical for Benign Keratosis-like Lesions', 'typical for Dermatofibroma', 'typical for Melanoma', 'typical for Melanocytic Nevus', 'typical for Vascular Lesions'],
    'dermoscopic patterns': ['typical for Actinic Keratoses', 'typical for Basal Cell Carcinoma', 'typical for Benign Keratosis-like Lesions', 'typical for Dermatofibroma', 'typical for Melanoma', 'typical for Melanocytic Nevus', 'typical for Vascular Lesions'],
    'texture': ['typical for Actinic Keratoses', 'typical for Basal Cell Carcinoma', 'typical for Benign Keratosis-like Lesions', 'typical for Dermatofibroma', 'typical for Melanoma', 'typical for Melanocytic Nevus', 'typical for Vascular Lesions'],
    'symmetry': ['typical for Actinic Keratoses', 'typical for Basal Cell Carcinoma', 'typical for Benign Keratosis-like Lesions', 'typical for Dermatofibroma', 'typical for Melanoma', 'typical for Melanocytic Nevus', 'typical for Vascular Lesions'],
    'elevation': ['typical for Actinic Keratoses', 'typical for Basal Cell Carcinoma', 'typical for Benign Keratosis-like Lesions', 'typical for Dermatofibroma', 'typical for Melanoma', 'typical for Melanocytic Nevus', 'typical for Vascular Lesions']

}

explicid_isic_dict_mine = {
    'asymmetry': ['none',   'none or partial', 'partial',  'complete'],
    'border irregularity': ['regular', 'regular or slightly irregular', 'slightly irregular', 'slightly or highly irregular','highly irregular'],
    'color variation': ['uniform', 'uniform or mild variation','mild variation', 'significant variation'],
    'diameter': ['small (<6mm) or moderate (6–20mm)',  'moderate (6–20mm)',  'large (>20mm)'],
    'texture': ['smooth', 'smooth or slightly rough','slightly rough', 'slightly rough or very rough',  'very rough'],
    'vascular patterns': ['none', 'optional','present (e.g., telangiectasia)'],
}
# 24
explicid_busi_dict = {
    'shape': ['round', 'oval', 'irregular'],
    'margin': ['circumscribed or indistinct', 'circumscribed or microlobulated', 'angular or spiculated'],
    'echo': ['anechoic or isoechoic', 'uypoechoic or isoechoic', 'hypoechoic or complex'],
    'posterior': ['none', 'enhancement or none', 'shadowing'],
    'calcifications': ['none', 'none or macrocalcifications','microcalcifications'],
    'orientation': ['parallel', 'non-parallel']

}

explicid_busi_soft_smooth_dict = {
    'shape': ['round', 'oval', 'irregular'],
    'margin': ['circumscribed', 'microlobulated', 'indistinct', 'spiculated'],
    'echo pattern': ['anechoic', 'hypoechoic', 'hyperechoic', 'complex'],
    'posterior features': ['enhancement', 'shadowing', 'none'],
    'calcifications': ['present','none'],
    'orientation': ['parallel', 'non-parallel']

}

explicid_idrid_dict = {
    'microaneurysms': ['none', 'few', 'moderate', 'many'],
    'hemorrhages': ['None', 'few', 'moderate', 'extensive'],
    'exudates': ['none', 'none or hard', 'hard or soft', 'hard or soft or mixed'],
    'neovascularization': ['none', 'present'],
    'macular edema': ['none', 'none or present in focal areas', 'present in focal areas','diffuse'],
}

explicid_idrid_soft_dict = {
    'hemorrhages': ['no hemorrhages', 'few microaneurysms', 'multiple dot and blot hemorrhages', 'large hemorrhages in all quadrants', 'widespread hemorrhages'],
    'microaneurysms': ['no microaneurysms', 'few microaneurysms', 'multiple scattered microaneurysms', 'numerous microaneurysms in all quadrants', 'many microaneurysms, often near neovascularization'],
    'exudates': ['no exudates', 'small hard exudates', 'moderate-sized exudates', 'large exudate patches', 'extensive exudates around macula'],
    'neovascularization': ['no neovascularization', 'neovascularization on optic disc or elsewhere','neovascularization extensive with risk of hemorrhage'],
    'venous beading': ['no venous beading', 'venous beading present in some quadrants','widespread and severe venous beading'],
    'cotton wool spots': ['no cotton wool spots', 'few cotton wool spots','multiple cotton wool spots', 'widespread cotton wool spots'],
    'macular edema': ['no macular edema', 'mild thickening of macular edema','moderate thickening of macular edema','severe thickening of macular edema with distortion ','severe thickening of macular edema with retinal detachment'],
}

explicid_idrid_edema_soft_dict = {
    'hard exudates near macula': ['no hard exudates near macula', 'few scattered hard exudates near macula', 'dense hard exudates near macula near macula'],
    'distance to fovea': ['retinal lesions are far from fovea', 'retinal lesions are close to fovea', 'retinal lesions are involving fovea'],
    'macular edema extent (approx)': ['no macular edema', 'focal (clustered) macular edema', 'diffuse (spread) macular edema'],
    'microaneurysms/hemorrhages near macula': ['no microaneurysms/hemorrhages near macula', 'few microaneurysms/hemorrhages near macula', 'many microaneurysms/hemorrhages near macula']
}

explicid_idrid_edema_dict = {
    'hard exudates': ['none','none or bright lipid deposits in the retina','bright lipid deposits in the retina'],
    'retinal thickening': ['none', 'swelling near the macula'],
    'microaneurysms': ['none', 'small bulges in blood vessels'],
    'hemorrhages': ['none', 'none or retinal bleeding','retinal bleeding'],
    'cotton wool spots': ['none', 'white patches due to nerve layer damage'],
    'vascular abnormalities': ['none', 'none or enlarged or irregular blood vessels', 'enlarged or irregular blood vessels']
}