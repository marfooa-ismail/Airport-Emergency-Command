# Airport Emergency Command Center v11 Final

Run with:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Final changes in v11:
- Space-to-runway emergency sequence before final approach.
- More realistic vector airplane with fuselage, wings, engines, windows, landing gear, and ground rollout.
- Aircraft aligns with runway centerline, touches down, rolls out, then taxis with rescue escort.
- ATC announcement starts first; ambulance and fire/rescue sirens are triggered after clearance/touchdown sequence.
- Morning/day/evening/night/rain/fog/thunderstorm visuals update inside the simulator.
- Other airlines are slowed and shown in holding pattern while emergency aircraft receives priority.
- Training dashboard uses visual graphs instead of raw JSON.


## Runtime model compatibility
If models are missing or incompatible with your Python/scikit-learn version, the app now automatically retrains them from the included dataset on first emergency launch. This fixes ModuleNotFoundError: No module named _loss.

## Fix for `_loss` / old model loading errors
If you see `ModuleNotFoundError: No module named '_loss'`, it means the bundled trained model files were created with a different scikit-learn/Python version. This version auto-deletes incompatible model files and retrains fresh models from the included dataset the first time you trigger an emergency or open training. You can also force retraining manually:

```bash
python scripts/train_models.py
```
