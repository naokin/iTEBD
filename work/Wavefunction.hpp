
template<typename T>
struct Wavefunction {

  //    +1  -1
  // +1 uu  ud
  // -1 du  dd

  MATRIX<T> uu_; ///< |Sz=+1,+1>

  MATRIX<T> ud_; ///< |Sz=+1,-1>

  MATRIX<T> du_; ///< |Sz=-1,+1>

  MATRIX<T> dd_; ///< |Sz=-1,-1>

  Wavefunction (const iMPS<T>& A, const iMPS<T>& B)
  {
    uu_ = A.u_ * B.u_;
    ud_ = A.u_ * B.d_;
    du_ = A.d_ * B.u_;
    dd_ = A.d_ * B.d_;
  }

  void decompose (iMPS<T>& A, iMPS<T>& B) const
  {
    // C => A x B
    const boundaryInfo& infoAB = *(A.r_);
    const boundaryInfo& infoBA = *(B.r_);

    // Assumed each matrix to be block-diagonalized
    for(size_t i = 0; i < infoAB.spin_z_.size(); ++i) {
      // Collect quantized tiles
      for(size_t j = 0; j < infoBA.spin_z_.size(); ++j) {
      }
    }
  }

};

