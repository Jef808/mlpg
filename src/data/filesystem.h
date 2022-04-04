#ifndef FILESYSTEM_H_
#define FILESYSTEM_H_

#include <utility>
#include <string_view>
#include <filesystem>


namespace simple::Data {

/**
 * Resolve symlinks and return paths relative to $HOME env variable.
 *
 * It is expected that inside the @data_dir, there is one train.csv file
 * and one test.csv file.
 */
std::pair< std::string, std::string >
get_train_test_fp(std::string_view data_dir, std::error_code& ec) noexcept {
    namespace fs = std::filesystem;

    fs::path data_dir_{ data_dir };
    data_dir_ = fs::is_symlink(data_dir_, ec)
        ? fs::read_symlink(data_dir_, ec)
        : data_dir_;
    if (ec)
        return {};

    const char* home = getenv("HOME");

    fs::path train_fp = fs::relative(data_dir_ / "train.csv", home);
    fs::path test_fp = fs::relative(data_dir_ / "test.csv", home);

    return std::make_pair( train_fp.string(), test_fp.string() );
}


}  // namespace simple::Data

#endif // FILESYSTEM_H_
