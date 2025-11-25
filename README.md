# Current prediction of satellite solar arrays

This project proposes an enhanced recursive multi-step current prediction method that incorporates sensor-driven operating mode perception. The contribution lies in leveraging probabilistic modeling to enhance temporal convolutional networks for accurate operating mode perception, matching appropriate predictor libraries, and proposing an attention-enhanced recursive framework for multi-step current prediction. A key advantage lies in the ability of attention-enhanced frameworks to adaptively perform multi-step current prediction on raw telemetry data, guided by the identified operating modes.


## Code architecture

The code consists of three parts, namely partial data of the geostationary satellite electrical power system, the code of the proposed method (main. py, perception. py, prediction. py), and comparative experiments (Direct_LSTM. py, Direct_TCN. py, DirRec-B3. py, GRU. py, Seq2Seq. py).


## Running Tests

To run the test, run the following command.

```bash
  python main.py
```
