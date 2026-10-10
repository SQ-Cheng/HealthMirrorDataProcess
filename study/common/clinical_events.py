"""Verified target-specific lab identities for distinct-measurement batches."""

import numpy as np
import pandas as pd


def attach_clinical_events(records, manifest, value_column):
    prefix = value_column.removesuffix("_value")
    source = manifest.set_index("video_id", verify_integrity=True).loc[records.video_id]
    times = pd.to_numeric(source[f"{prefix}_lab_time_unix"], errors="raise").to_numpy(float)
    if not np.array_equal(source.hospital_id.to_numpy(), records.hospital_id.to_numpy()) or not np.isfinite(times).all():
        raise ValueError("Matched laboratory event identity/time is invalid")
    np.testing.assert_allclose(source[value_column], records.raw_value, rtol=0, atol=1e-9)
    result = records.copy()
    result["label_time_unix"] = times
    result["clinical_event_id"] = result.hospital_id.astype(str) + "@" + pd.Series(times,index=result.index).map(lambda x: format(x,".17g"))
    if result.groupby("clinical_event_id")[["raw_value","binary_label"]].nunique().gt(1).any().any():
        raise ValueError("One clinical measurement has inconsistent values/labels")
    return result
