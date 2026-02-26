import subprocess
import re

def git_branch(path='.'):
    try:
        out = subprocess.check_output(['git', '-C', path, 'rev-parse', '--abbrev-ref', 'HEAD'],
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except subprocess.CalledProcessError:
        return None  # not a git repo or error

def git_remote_url(path='.', remote='origin'):
    try:
        out = subprocess.check_output(['git', '-C', path, 'remote', 'get-url', remote],
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except subprocess.CalledProcessError:
        return None

def parse_github_owner_repo(url):
    # handles formats like:
    #   git@github.com:owner/repo.git
    #   https://github.com/owner/repo.git
    if not url:
        return (None, None)
    m = re.search(r'[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$', url)
    if m:
        return m.group('owner'), m.group('repo')
    return (None, url)  # fallback: return entire url as "repo"

if __name__ == '__main__':
    branch = git_branch('.')
    url = git_remote_url('.')
    owner, repo = parse_github_owner_repo(url)
    print('branch:', branch)
    print('owner:', owner)
    print('repo:', repo)
    # or combined:
    print(f'{owner}/{repo}:{branch}')