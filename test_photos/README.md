# Test photos

Photos for testing the vision model, OCR and aggregation by hand. Not used for training
(the training dataset goes in `dataset/<object>/`).

Name each photo `<object>_<short_description>.<ext>`, where `<object>` is one of
`door_handle`, `cabinet_door`, `chair_table`, `drawer`, `wall_shelf`, and add a row to
`expected.json` with what the app should find. This turns the folder into a small test set.

| File | Object | Expected OCR |
|---|---|---|
| drawer_ikea_label.png | drawer | IKEA MALM, Art. 403.604.65 |
