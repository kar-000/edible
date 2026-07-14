export function AboutPage() {
  return (
    <div className="about-page">

      <section className="about-section">
        <h2>How identification works</h2>
        <p>
          Every photo passes through three sequential checks. If any check fails, the image
          is rejected rather than guessed at — a false rejection is always safer than a false
          identification.
        </p>

        <ol className="about-pipeline">
          <li>
            <div className="pipeline-step">
              <span className="pipeline-num">1</span>
              <div>
                <strong>Plant gate</strong>
                <p>A MobileNetV3 image classifier confirms the photo contains a plant.
                Non-plant images (animals, landscapes, objects) are rejected immediately.</p>
              </div>
            </div>
          </li>
          <li>
            <div className="pipeline-step">
              <span className="pipeline-num">2</span>
              <div>
                <strong>Fruit presence gate</strong>
                <p>A CLIP zero-shot model checks that fruit or berries are actually
                visible. Bark, leaves, and roots alone don't contain enough information
                for safe identification.</p>
              </div>
            </div>
          </li>
          <li>
            <div className="pipeline-step">
              <span className="pipeline-num">3</span>
              <div>
                <strong>Species classifier</strong>
                <p>A DINOv2 ViT-B/14 vision transformer (86M parameters, fine-tuned on
                our Texas berry dataset) predicts which of 12 species is shown. The model
                was trained to heavily penalise mistaking a toxic species for an edible
                one — that class of error is treated as far worse than a rejection.</p>
              </div>
            </div>
          </li>
        </ol>
      </section>

      <section className="about-section">
        <h2>Confidence score</h2>
        <p>
          The raw model output is a probability over 12 species. We apply <strong>temperature
          scaling</strong> (T&nbsp;=&nbsp;0.80) to calibrate those probabilities — uncalibrated
          neural networks tend to be overconfident, and temperature scaling corrects for that.
        </p>
        <div className="about-threshold-box">
          <div className="threshold-row threshold-row--safe">
            <span className="threshold-pct">≥ 85%</span>
            <span>High confidence — green bar</span>
          </div>
          <div className="threshold-row threshold-row--warn">
            <span className="threshold-pct">75 – 84%</span>
            <span>Moderate confidence — amber bar. Treat with extra caution.</span>
          </div>
          <div className="threshold-row threshold-row--danger">
            <span className="threshold-pct">&lt; 75%</span>
            <span>Below safety floor — <strong>rejected</strong>. Do not eat.</span>
          </div>
        </div>
      </section>

      <section className="about-section">
        <h2>Location &amp; range checking</h2>
        <p>
          When you share your location, we check whether the identified species is documented
          in your Texas county using USDA PLANTS distribution data. If the species has no
          confirmed records in your county, the confidence score is multiplied by 0.5× before
          the 75% floor check — this means an 80% match becomes 40% and is rejected.
        </p>
        <p>
          This is a <em>down-ranking</em>, not a hard block. A species outside its typical
          range is less likely but not impossible (range data has gaps, and species move).
          Geocoding is handled by OpenStreetMap Nominatim; if it fails, no penalty is applied.
        </p>
      </section>

      <section className="about-section">
        <h2>Scope</h2>
        <ul className="about-list">
          <li><strong>Geography:</strong> Texas statewide (Phase 1). Central Texas has the
          densest training data due to iNaturalist observation density.</li>
          <li><strong>Species:</strong> 12 species — 6 edible, 6 toxic. See the full list below.</li>
          <li><strong>Season:</strong> Only fruiting-stage images were used for training.
          Identification from leaves or bark alone is not supported.</li>
        </ul>

        <div className="species-grid">
          <div className="species-col">
            <h3 className="species-col__heading species-col__heading--edible">Edible species</h3>
            <ul>
              <li>Beautyberry <em>(Callicarpa americana)</em></li>
              <li>Sugarberry <em>(Celtis laevigata)</em></li>
              <li>Agarita <em>(Mahonia trifoliolata)</em></li>
              <li>Dewberry <em>(Rubus trivialis)</em></li>
              <li>Elderberry <em>(Sambucus canadensis)</em></li>
              <li>Mustang grape <em>(Vitis mustangensis)</em></li>
            </ul>
          </div>
          <div className="species-col">
            <h3 className="species-col__heading species-col__heading--toxic">Toxic species</h3>
            <ul>
              <li>Possumhaw <em>(Ilex decidua)</em></li>
              <li>Yaupon holly <em>(Ilex vomitoria)</em></li>
              <li>Chinaberry <em>(Melia azedarach)</em></li>
              <li>Pokeweed <em>(Phytolacca americana)</em></li>
              <li>Black nightshade <em>(Solanum nigrum)</em></li>
              <li>Carolina horsenettle <em>(Solanum carolinense)</em></li>
            </ul>
          </div>
        </div>
      </section>

      <section className="about-section">
        <h2>Data sources</h2>
        <ul className="about-sources">
          <li>
            <strong>Training images</strong> — iNaturalist community observations,
            licensed CC BY-NC 4.0. Only research-grade observations with fruiting
            phenology were included. Images are not redistributed.
          </li>
          <li>
            <strong>County range data</strong> — USDA PLANTS Database
            (plants.usda.gov), public domain. Distribution records for Texas counties.
          </li>
          <li>
            <strong>Geocoding</strong> — OpenStreetMap Nominatim, data ©
            OpenStreetMap contributors (ODbL).
          </li>
          <li>
            <strong>Model backbone</strong> — DINOv2 ViT-B/14 by Meta AI,
            Apache 2.0 license.
          </li>
          <li>
            <strong>Fruit gate</strong> — CLIP ViT-B/32 by OpenAI, used
            zero-shot for fruit presence detection.
          </li>
          <li>
            <strong>Plant gate</strong> — MobileNetV3, pretrained on ImageNet.
          </li>
        </ul>
      </section>

      <section className="about-section about-section--disclaimer">
        <h2>Important disclaimer</h2>
        <p>
          This app is an <strong>educational tool only</strong>. It is not a substitute
          for expert botanical knowledge. Never eat a wild plant based solely on an app
          identification. Misidentification of wild plants can cause serious illness or
          death. When in doubt, do not eat it.
        </p>
        <p>
          Look-alike warnings are provided for the most dangerous confusable species but
          are not exhaustive. Local variations in plant appearance may not be captured by
          the training data.
        </p>
      </section>

    </div>
  )
}
