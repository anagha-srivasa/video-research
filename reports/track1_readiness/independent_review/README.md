# Independent review pack

Status: pending human annotation; zero validated references.

Complete A and B separately, blinded to model outputs. Start with target_definition worksheets: class cues locate the intended entity, but are provisional source labels. Agree an observable physical landmark through independent definitions and adjudication; do not treat a changing mask centroid as a physical anchor. Exclude ambiguous or invisible targets.

Anchor sheets reference actual native frames by uncompressed NPY path and storage row. Use `numpy.load(path, mmap_mode="r")[storage_row]` to inspect native RGB. Do not rescale coordinates. RGB hashes are in rgb_manifest.json; submitted annotations must include reviewer identity and file hashes.

Contact requires independently traced native-pixel boundaries and sign/distance conventions; identity and events require episode-level review. Empty fields are pending, not negative labels. Agreement and reliability must be measured before derivative/event threshold selection. These development clips cannot establish fresh final acceptance.
