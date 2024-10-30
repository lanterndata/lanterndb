import argparse
import subprocess
import getpass
import git
import os

# placeholder used in sql update scripts as the next release version
LATEST="latest"

class Version:
    def __init__(self, version: str):
        self.latest = False
        self.version = version
        if version == LATEST:
            self.latest = True
            return

        self.version_numbers = [int(n) for n in version.split('.')]
    def __lt__(self, other):
        if self.latest:
            return False
        if other.latest:
            return True
        for i, v in enumerate(self.version_numbers):
            if v < other.version_numbers[i]:
                return True
            if v > other.version_numbers[i]:
                break
        return False

    def __eq__(self, other):
        if self.latest or other.latest:
            return self.latest == other.latest
        for i, v in enumerate(self.version_numbers):
            if v != other.version_numbers[i]:
                return False
        return True
    def __le__(self, other):
        return self < other or self == other
    def __ne__(self, other):
        return not self == other
    def __gt__(self, other):
        return not self == other and not self < other
    def __ge__(self, other):
        return not self < other
    def __str__(self):
        return self.version
    def __repr__(self):
        return self.version

INCOMPATIBLE_VERSIONS = {
    '16': [Version('0.0.4')],
    '17': [Version('0.3.0'), Version('0.3.1'), Version('0.3.2'), Version('0.3.3'), Version('0.3.4'), Version('0.4.0'), Version('0.4.1')],
}

def shell(cmd, exit_on_error=True):
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        if res.stderr:
                print("Error building from tag" + res.stderr)
        print("res stdout", res.stdout, res)
        if exit_on_error:
            exit(1)
        else:
            print("ERROR on command", cmd)


# Make sure lantern can smoothly be updated from from_version to to_version
# the function installs the DB at from_version, runs an upgrade via ALTER EXTENSION ... UPDATE
# and runs the test suit on the resulting DB
# Note: from_version must be a valid tag on the repo that has a corresponding release and SQL migration script
# to_version must be the value LATEST or follow the requirements above
def update_from_tag(from_version: str, to_version: str, starting_point  = None):
    from_tag = "v" + from_version
    repo = git.Repo(search_parent_directories=True)
    print(repo.remotes)
    starting_sha = starting_point if starting_point else repo.head.object.hexsha
    to_sha = starting_sha

    if to_version != LATEST:
        to_tag = "v" + to_version
        tag_names = [tag.name for tag in repo.tags]
        if to_tag in tag_names:
            to_sha = to_tag
        else:
            print(f"WARNING: to_version=${to_version} has not corresponding tag. assuming current HEAD corresponds to that version")

    try:
        repo.remotes[0].fetch()
    except Exception as e:
        # fetching does not work in the dev dockerfile but it does not need to,
        # since we are testing the updates on the local repo
        if not "error: cannot run ssh" in str(e):
            raise Exception(f"unknown fetch error: {e}")

    rootdir = args.rootdir
    if Version(from_version) < Version("0.4.0"):
        rootdir = ".."

    sha_after = repo.head.object.hexsha
    print(f"Updating from tag {from_tag}(sha: {sha_after}) to {to_version}")

    # check out to the old version only for binary and catalog update script installation.
    # checkout to latest to make sure we always run the latest version of all scripts
    repo.git.checkout(from_tag)
    # run "mkdir build && cd build && cmake .. && make -j4 && make install"
    res = shell(f"rm -rf {args.builddir} || true")
    res = shell(f"mkdir -p {args.builddir} ; git submodule update --init --recursive && cmake -DRELEASE_ID={from_version} -S {rootdir} -B {args.builddir} && make -C {args.builddir} -j install")
    repo.git.checkout(starting_sha)
    # We are just compilinig again (not installing)
    # Because file structure was changed afte 0.4.0 version
    # And cmake complains that CMakeFiles does not exist
    res = shell(f"rm -rf {args.builddir} third_party && mkdir -p {args.builddir} && git submodule update --init --recursive && cmake -DRELEASE_ID={from_version} -S {args.rootdir} -B {args.builddir} && make -C {args.builddir} -j")


    res = shell(f"psql postgres -U {args.user} -c 'DROP DATABASE IF EXISTS {args.db};'")
    res = shell(f"psql postgres -U {args.user} -c 'CREATE DATABASE {args.db};'")
    res = shell(f"psql postgres -U {args.user} -c 'DROP EXTENSION IF EXISTS lantern CASCADE; CREATE EXTENSION lantern;' -d {args.db};")

    # run begin of parallel tests. Run this while the from_tag version of the binary is installed and loaded run begin on {from_version}
    if from_tag == "v0.0.4":
        # the source code at 0.0.4 did not yet have parallel tests
        return

    # remove temporary files generated by test runner
    res = shell('rm -f /tmp/ldb_update.lock')
    res = shell('rm -f /tmp/ldb_update_finished')
    res = shell(f"cd {args.builddir} ; UPDATE_EXTENSION=1 UPDATE_FROM={from_version} UPDATE_TO={from_version} make test-parallel FILTER=begin")

    res = shell(f"cd {args.builddir} ; UPDATE_EXTENSION=1 UPDATE_FROM={from_version} UPDATE_TO={from_version} make test-misc FILTER=begin")

    repo.git.checkout(to_sha)
    res = shell(f"rm -rf {args.builddir} third_party && rm -rf third_party && mkdir -p {args.builddir} && git submodule update --init --recursive && cmake -DRELEASE_ID={to_version} -S {args.rootdir} -B {args.builddir} && cd {args.builddir} && make -j install")
    repo.git.checkout(starting_sha)

    res = shell(f"cd {args.builddir} ; UPDATE_EXTENSION=1 UPDATE_FROM={from_version} UPDATE_TO={from_version} make test-misc FILTER=version_mismatch")

    res = shell(f"cd {args.builddir} ; UPDATE_EXTENSION=1 UPDATE_FROM={from_version} UPDATE_TO={to_version} make test")
    # run the actual parallel tests after the upgrade
    res = shell('rm -f /tmp/ldb_update.lock')
    res = shell('rm -f /tmp/ldb_update_finished')
    res = shell(f"cd {args.builddir} ; UPDATE_EXTENSION=1 UPDATE_FROM={from_version} UPDATE_TO={to_version} make test-parallel EXCLUDE=begin")

    print(f"Update {from_version}->{to_version} Success!")

def try_update_from_tag(from_version, to_version):
    repo = git.Repo(search_parent_directories=True)
    starting_sha = None
    try:
        starting_sha = repo.active_branch.name
    except Exception as e:
        print(f"Did not detect active branch: {e}. Using HEAD as starting point")
        starting_sha = repo.head.object.hexsha

    try:
        update_from_tag(from_version, to_version, starting_sha)
    except Exception as e:
        repo.git.checkout(starting_sha)
        print(f"Error updating from {from_version} to {to_version}: {e}")

def incompatible_version(pg_version, version_tag):
    if not pg_version or pg_version not in INCOMPATIBLE_VERSIONS:
        return False
    return version_tag in INCOMPATIBLE_VERSIONS[pg_version]


if __name__ == "__main__":

    default_user = getpass.getuser()

    # collect the tag from command line to upgrade from

    parser = argparse.ArgumentParser(description='Update from tag')
    parser.add_argument('-from_tag', '--from_tag', metavar='from_tag', type=str,
                        help='Tag to update from', required=False)
    parser.add_argument('-to_tag','--to_tag', metavar='to_tag', type=str,
                        help='Tag to update to', required=False)
    parser.add_argument("-db", "--db", default="update_db", type=str, help="Database name used for updates")
    parser.add_argument("-U", "--user",  default=default_user, help="Database user")
    parser.add_argument("-builddir", "--builddir",  default="build_updates", help="Build directory")
    parser.add_argument("-rootdir", "--rootdir",  default=".", help="Root directory")

    args = parser.parse_args()

    from_tag = args.from_tag
    to_tag = args.to_tag
    if from_tag and to_tag:
        try_update_from_tag(from_tag, to_tag)
        exit(0)

    if from_tag or to_tag:
        print("Must specify both or neither from_tag and to_tag")
        exit(1)

    # test updates from all tags
    version_pairs = [update_fname.split("--") for update_fname in os.listdir(f"{args.rootdir}/sql/updates")]
    version_pairs = [(from_version, to_version.split('.sql')[0]) for from_version, to_version in version_pairs]
    repo = git.Repo(search_parent_directories=True)
    tags_actual = [tag.name for tag in repo.tags]
    tags_actual = [name[1:] if name[0] == 'v' else name for name in tags_actual]

    version_pairs = [(from_v, to_v) for from_v, to_v in version_pairs]
    from_versions = list(sorted([Version(p[0]) for p in version_pairs]))
    from_versions.reverse()
    to_versions = list(sorted([Version(p[1]) for p in version_pairs]))
    for from_v in from_versions:
        assert(str(from_v) in tags_actual)

    num_untagged = 0
    for to_v in to_versions:
        # only the last to_v may be untagged (when the release has not happened yet)
        if str(to_v) not in tags_actual:
            assert(num_untagged == 0 and "ERROR: found second untagged version. At most one (the latest) untagged version expected")
            num_untagged += 1
            if num_untagged != 0:
                print(f"WARNING: version {to_v} not in repo tags ${tags_actual}")

    if len(to_versions) > 0:
        latest_version = to_versions[-1]
        print("Updating from tags", from_versions, "to ", latest_version)

        pg_version = None if not 'PG_VERSION' in os.environ else os.environ['PG_VERSION']
        for from_tag in from_versions:
            if incompatible_version(pg_version, from_tag):
                continue
            try_update_from_tag(str(from_tag), str(latest_version))
