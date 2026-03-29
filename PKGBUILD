# Maintainer: lasi
pkgname=gpu-compressor
pkgver=0.1.0
pkgrel=1
pkgdesc="GPU-accelerated file compression using NVIDIA nvCOMP"
arch=('x86_64')
url="https://github.com/Lasimeri/gpu-compressor"
license=('MIT')
depends=('cuda' 'nvcomp')
makedepends=('rust' 'clang')
source=("$pkgname-$pkgver.tar.gz::$url/archive/v$pkgver.tar.gz")
sha256sums=('SKIP')

build() {
    cd "$pkgname-$pkgver"
    cargo build --release --locked
}

package() {
    cd "$pkgname-$pkgver"
    install -Dm755 "target/release/$pkgname" "$pkgdir/usr/bin/$pkgname"
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
    install -Dm644 README.md "$pkgdir/usr/share/doc/$pkgname/README.md"
}
