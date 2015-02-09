

struct boundaryInfo {

  std::vector<int> spin_z_;

  std::vector<int> extent_;

  std::vector<double> lambda_;

  /// Constructor
  /// \param mxz max 2*Sz value, i.e. spin quantum numbers { -Sz, -Sz+1, ... , Sz-1, Sz } are generated.
  /// \param mxD max size of each dense block
  boundaryInfo (int mxz, int mxD);
};
