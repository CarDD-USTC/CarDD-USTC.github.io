template <typename T>
MOBULA_KERNEL cumsum_kernel(const int N, const T* X, T* I, const int att_size) {
  parfor(N, [&](int b) {
    const T* Xi = X + b * att_size;
    T* Ii = I + b * att_size;
    for (int i = 1; i < att_size; ++i) {
      Ii[i] = Ii[i - 1] + Xi[i];
    }
  });
}

template <typename T>
MOBULA_KERNEL map_step_kernel(const int N, const T* attxi, T* index_x,
                              const T* stepxs, const int att_size,
                              const int out_size) {
  T myscale = T(2) / (att_size - 1);
  parfor(N, [&](int b) {
    int i = 0;
    int j = 0;
    const T* mapxi = attxi + b * att_size;
    const T stepx = stepxs[b];
    T* index_i = index_x + b * out_size;
    while (i < out_size && j < att_size) {
      if (mapxi[j] >= i * stepx) {
        index_i[i] = (j + (i * stepx - mapxi[j]) /
                              (mapxi[j] - (j >= 1 ? mapxi[j - 1] : 0))) *
                         myscale -
                     1.0;
        i++;
      } else {
        j++;
      }
    }
    if (i < out_size) {
      T value = i == 0 ? 0 : index_i[i - 1];
      for (; i < out_size; ++i) index_i[i] = value;
    }
  });
}
