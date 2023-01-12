Code for the paper [Efficient Human-in-the-loop System for Guiding DNNs Attention](https://arxiv.org/abs/2206.05981)

## Start

- Prepare the [User Interface](https://github.com/ultratykis/User-Interface-For-Guiding-DNNs-Attention.git).
- Prepare the dataset with the following structure.
  ```
  dataset/
      <dataset_name>/
          <train>/
          <test>/
          <val>/
  ```
- Set up the environment.
  ```bash
  conda env create -f=env.yml
  ```
- Run the app.
  ```
  python app.py
  ```
