export const metadata = { title: "About · KidneyScan AI" };

export default function About() {
  return (
    <div className="ks-wrap ks-page ks-prose-page">
      <h1>About KidneyScan AI</h1>
      <p className="ks-lede">
        KidneyScan AI is a research prototype from Stochastic Research Labs for exploring automated kidney tumor and cyst segmentation on
        CT scans. It is meant for students learning CT segmentation and for small teams without a kidney specialist.
      </p>

      <h2>The data</h2>
      <p>
        <a href="https://kits-challenge.org/kits23/" target="_blank" rel="noreferrer">KiTS23</a> provides 599 cases: 489 public training cases and
        110 test cases with private masks. Every voxel is labelled background, kidney, tumor or cyst. The data is licensed CC BY-NC-SA 4.0: free
        for research, not commercial. It is not stored in this project.
      </p>

      <h2>The model</h2>
      <p>
        A light 3D U-Net with 1.68M parameters reads a whole CT volume and labels each voxel. It is far smaller than the ensembles used in
        challenge entries, which is the point: it is meant to be easy to run and easy to study. The results are modest, and the{" "}
        <a href="/insights">model insights</a> page shows them as they are.
      </p>

      <h2>The viewer</h2>
      <p>
        Your scan is read in the browser and drawn with Three.js by ray-marching through the volume. The original CT and the segmentation
        overlay share one camera, so you can rotate a full 360 degrees and compare the same angle. Volumes in cm³ are counted from the
        predicted mask using the voxel size stored in the scan.
      </p>

      <h2>Limits</h2>
      <p>
        This is not a medical device. Predictions can be wrong, especially for tumors and cysts, and the model gives no per-case accuracy. A
        doctor must confirm any finding. The home page kidney is an illustration generated in code, not a scan or a result.
      </p>
    </div>
  );
}
