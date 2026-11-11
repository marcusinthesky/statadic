select path, coalesce(json_extract(frontmatter, '$.title'), title, stem) as label,
       coalesce(json_extract(frontmatter, '$.description'), '') as description,
       coalesce(json_extract(frontmatter, '$.status'), '') as status,
       case when json_extract(frontmatter, '$.status') in ('done', 'closed')
            then 1 else 0 end as terminal
from documents
where json_extract(frontmatter, '$.type') in ('plan', 'task')
  and (
    (:insitu_path = 'ROADMAP.md' and parent like '.context/plan/%'
       and instr(substr(parent, length('.context/plan/') + 1), '/') = 0)
    or
    (:insitu_path != 'ROADMAP.md' and parent like :insitu_dir || '/tasks/%'
       and instr(substr(parent, length(:insitu_dir) + 8), '/') = 0)
  )
order by path;
