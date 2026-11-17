import DiceTable from "../../components/DiceTable";
import { MODEL_FACTS } from "../../lib/metrics";

export const metadata = { title: "Model insights · KidneyScan AI" };

export default function Insights() {
  return (
    <div className="ks-wrap ks-page">
      <h1>Model performance</h1>
      <p className="ks-lede">
        Best results so far for the light 3D U-Net. These are honest numbers from a small model, and they leave plenty of room to improve.
      </p>

      <div className="ks-split ks-split-top">
        <section className="ks-card">
          <h2>Dice score by structure</h2>
          <DiceTable />
          <p className="ks-small">
            Dice measures overlap between the prediction and the expert outline. 1 is a perfect match and 0 is no overlap. The foreground
            score is the average of the three structure scores.
          </p>
        </section>

        <section className="ks-card">
          <h2>Reading the numbers</h2>
          <ul className="ks-plain">
            <li>
              <b>Kidney (0.787).</b> The clearest result. Still expect boundary errors.
            </li>
            <li>
              <b>Tumor (0.378).</b> The model finds part of the tumor. Do not rely on its outline or its volume.
            </li>
            <li>
              <b>Cyst (0.048).</b> Close to failing. Cysts are small and uncommon, and they are the hardest class. Treat any cyst result as
              unreliable.
            </li>
          </ul>
          <p className="ks-small">Volumes shown in the workspace come from the predicted mask, so they inherit these errors.</p>
        </section>
      </div>

      <section className="ks-card ks-facts">
        <h2>Model and data</h2>
        <dl>
          <div><dt>Architecture</dt><dd>{MODEL_FACTS.architecture}</dd></div>
          <div><dt>Parameters</dt><dd>{MODEL_FACTS.parameters}</dd></div>
          <div><dt>Dataset</dt><dd>{MODEL_FACTS.dataset}, {MODEL_FACTS.cases} cases ({MODEL_FACTS.trainingCases} public training, {MODEL_FACTS.testCases} test with private masks)</dd></div>
          <div><dt>Labels</dt><dd>0 background, 1 kidney, 2 tumor, 3 cyst</dd></div>
          <div><dt>Input</dt><dd>One 3D CT volume (.nii.gz)</dd></div>
        </dl>
      </section>
    </div>
  );
}
