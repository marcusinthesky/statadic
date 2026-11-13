select path, coalesce(json_extract(frontmatter, '$.title'), title, stem) as label,
       coalesce(json_extract(frontmatter, '$.description'), '') as description
from documents
where parent like :insitu_dir || '/%'
  and instr(substr(parent, length(:insitu_dir) + 2), '/') = 0
order by label, path;
