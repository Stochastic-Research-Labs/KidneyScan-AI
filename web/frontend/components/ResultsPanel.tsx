import { LABELS } from "../lib/labels";
import type { ClassScores, ClassVolumes } from "../lib/types";

type Props = {
  volumes: ClassVolumes | null;
  scores: ClassScores | null;
  isDummy: boolean;
};

export default function ResultsPanel({ volumes, scores, isDummy }: Props) {
  return (
    <section className="ks-card ks-results" aria-labelledby="results-title">
      <h2 id="results-title">Predicted structures</h2>
      {!volumes ? (
        <p className="ks-muted">Volumes and scores appear here after segmentation.</p>
      ) : (
        <>
          <table className="ks-table">
            <thead>
              <tr>
                <th scope="col">Structure</th>
                <th scope="col">Volume</th>
                <th scope="col">Model score</th>
              </tr>
            </thead>
            <tbody>
              {LABELS.map((l) => {
                const v = volumes[l.key];
                const s = scores?.[l.key];
                return (
                  <tr key={l.key}>
                    <th scope="row">
                      <i style={{ background: l.color }} aria-hidden />
                      {l.name}
                    </th>
                    <td>
                      {v > 0 ? (
                        <>
                          <b>{v.toFixed(1)}</b> cm³
                        </>
                      ) : (
                        <span className="ks-muted">None predicted</span>
                      )}
                    </td>
                    <td>{s == null ? <span className="ks-muted">{isDummy ? "n/a" : "Not returned"}</span> : s.toFixed(3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="ks-small">
            {isDummy
              ? "Dummy mode does not run the model, so there are no scores. Volumes are those of the synthetic shapes."
              : "Volumes are counted from the predicted mask using the scan's voxel size. The model score is what the server returns for each class. It is not a per-case accuracy measure and not a diagnosis."}
          </p>
        </>
      )}
    </section>
  );
}
